# (c) I. Marti-Vidal (Univ. Valencia) 2020.

# Script to simulate full-polarization pol-converted EU-VGOS observations.
# Due to limitations in the "sm" tool (tested on CASA 5.6), the simulation
# will be done using one single spectral window with 64 channels in it. 
# Then, we will use the "mstransform" task to split the channels in 
# independent spectral windows.

# This script complements the contents of the paper:

# I. Marti-Vidal et al. 2021, A&A, 646, 52



from __future__ import print_function

import numpy as np
import pylab as pl
import os, sys
import datetime as dt
import pickle as pk


# Seed of the random number generator (to get the same values as in the
# paper):
np.random.seed(42)

# Name of the final measurement set:
vis = 'EU-VGOS_Simul_1_py2.ms'


# Frequency configuration:
LO = 4.e9 # minimum frequency
nchan = 1 # channels per spw (in final dataset).
NIF = 64  # Number of IFs (aka "spws" in CASA jargon).
totwidth = 2.e9 # Total bandwidth of the observations.
width=totwidth/(NIF-1) # Bandwidth per IF.
BBs = [0.0] # Basebands (only one, with 2GHz width).

DUR = 32. # Scan duration (seconds). 


AMP = 0.09 # Maximum Dterm amplitude (for a 10 deg. X-Y offset; see Goddi et al. (2019)).
FARADAY = 1.e2 # Maximum rotation measure (in absolute value).
SPIX = 1.0  # Maximum spectral index (in absolute value).


# Random Dterms (pure imaginary and DR=DL, as expected from polconversion residuals):
DR = [[AMP*np.cos(np.linspace(0.,1.,NIF)*3.*np.pi)*1.j],
      [AMP*np.cos(0.30*np.pi+(np.linspace(0.,1.,NIF))*4.*np.pi)*1.j],
      [AMP*np.cos(0.65*np.pi+(np.linspace(0.,1.,NIF))*5.*np.pi)*1.j]]


Mon = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']



###################
## PREPARE SOURCE INFORMATION:

sl = open('VGOS_sources.dat')
sous = []
for line in sl.readlines():
  if len(line)>7 and line[0]!='#':


# Read source coordinates and observing times (parsed from a real EU-VGOS Vex file):
    J2000 = 'J2000 %sh%sm00.00 %sd%02im00.00'%(line[:2],line[2:4],line[4:7],int(line[7])*6)
    scans = line.split()[1:]
    year = int(scans[0][:4])
    dt0 = dt.date(year,1,1)

# Generate random values for this source's polarization:
    Pfr = 0.01 + 0.09*np.random.random()
    EVPA = np.random.random()*2.*np.pi
    spix = 2.*SPIX*np.random.random()-SPIX
    RM = np.random.random()*2.*FARADAY-FARADAY



#### Write information in CASA "listobs" format:
    scanTimes = []
    for scan in scans:
      day=int(scan[5:8])
      hh = scan[9:11]; mm = scan[12:14]; ss = scan[15:17]
      dt1 = dt0+dt.timedelta(day-1)
      dtstr = '%02i-%s-%i/%s:%s:%s.0'%(dt1.day,Mon[dt1.month-1],dt1.year,hh,mm,ss)
      h0 = float(hh)+float(mm)/60.+float(ss)/3600. + DUR/3600.
      hh1 = int(h0) 
      mm1 = (h0-hh1)*60.
      ss1 = (mm1 - int(mm1))*60.
      dtstr2 = '   -   %02i:%02i:%02i.0'%(hh1,int(mm1),int(ss1))
      scanTimes.append(dtstr+dtstr2)

    Ti = scanTimes[0].split()[0]
    Tf = scanTimes[-1].split('/')[0]+'/'+scanTimes[-1].split()[-1]
    ObsLine = 'Observed from   %s   to   %s (UTC)'%(Ti,Tf)
    sous.append([line.split()[0],J2000, Pfr*np.cos(2.*EVPA), Pfr*np.sin(2.*EVPA),spix,RM,ObsLine,scanTimes])

sl.close()

######################




# Create one measurement set per source (with polsimulate):

simvis = []
for sou in sous:

  print('\n\n SIMULATING %s\n\n'%sou[0])

  os.system('rm -rf %s_%s'%(vis,sou[0]))
  phCent = 'J2000 %02ih%02im00.00 %+02id%02im00.00'

########
# Create a temporary "listobs" file for this source:
  lobs = open('temp.listobs_%s'%sou[0],'w')
  print(sou[6],file=lobs)
  for k,sci in enumerate(sou[7]):
    print('%s  %i  0  %s  1  [0]  [1] [#ONSOURCE]'%(sci,k+1,sou[0]),file=lobs)
  lobs.close()
#######

# SIMULATE!
  if True:
    simvis.append('%s_%s'%(vis,sou[0]))
    polsimulate(vis='%s_%s'%(vis,sou[0]), 
            reuse=False, 
            array_configuration='VLBA.dat', # We call the array "VLBA". Otherwise, the 
#                                             CASA dictator will have problems.
            model_Dt_0 = [list(dii) for dii in DR], # Dterms for the first polarizer ("R")
            model_Dt_1 = [list(dii) for dii in DR], # Dterms for the second polarizer ("L")
            feed='circular', # We simulate R-L antenna feeds. 
            LO=LO, BBs=BBs, spw_width=totwidth, nchan=NIF, # Frequency configuration
            phase_center = sou[1], visib_time='1s', # Source coordinates and VLBI integration time.
            I = [1.0], Q_frac = [sou[2]], U_frac=[sou[3]], V_frac=[0.0], RM=[sou[5]], # source polarization.
            spec_index = [sou[4]], 
            RAoffset=[0.0], Decoffset=[0.0], # Source model is a centered point source.
            nscan='temp.listobs_%s'%sou[0], # If "nscan" is the path to a "listobs" file, polsimulate
#                                             uses the exact observing times from that file.
            onsource_time=DUR/3600.,observe_time=DUR/3600., # This is overriden by the "listobs" content.
            apply_parang=True, # Parallactic angle will be applied to the data
            export_uvf=False, # We do not need to export to uvfits (all is done in CASA)
            corrupt=True) # Will add noise, based on the default values (Trec=50K).

###########
# Polsimulate gives the name "polsim" to the source. We change it to its true name:
    tb.open('%s_%s/FIELD'%(vis,sou[0]),nomodify=False)
    NAM = tb.getcol('NAME')
    NAM[0] = sou[0]
    tb.putcol('NAME',NAM)
    tb.close()

    tb.open('%s_%s/SOURCE'%(vis,sou[0]),nomodify=False)
    NAM = tb.getcol('NAME')
    NAM[0] = sou[0]
    tb.putcol('NAME',NAM)
    tb.close()
############



# We now CONCATENATE all sources in one single Measurement Set:
os.system('rm -rf %s'%vis)
concat(vis=simvis, concatvis=vis,timesort=True)

# And we split each frequency channel into an independent IF:
os.system('rm -rf hola.ms')
mstransform(vis=vis, regridms=True,nspw=NIF,outputvis='hola.ms', datacolumn='data')
os.system('rm -rf %s_IF'%vis)
os.system('mv hola.ms %s_IF'%vis)


# Code not needed in CASA 5.6 and (hopefully) higher versions.
if False:
  tb.open(vis+'/DATA_DESCRIPTION',nomodify=False)
  spwi = tb.getcol('SPECTRAL_WINDOW_ID')
  polid = tb.getcol('POLARIZATION_ID')
  polid[:] = polid[0]
  tb.putcol('POLARIZATION_ID',polid)
  tb.close()
  tb.open(vis,nomodify=False)
  dd = tb.getcol('DATA_DESC_ID')
  dd[:] = 0
  tb.putcol('DATA_DESC_ID',dd)
  tb.close()


# We write the simulated source quantities into an external file,
# so we can make correlation plots (to polsolve fits) later:
off = open('simulation_values_1_py2.dat','w')
pk.dump([DR,sous],off)
off.close()
