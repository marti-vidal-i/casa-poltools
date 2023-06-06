# (c) I. Marti-Vidal (Univ. Valencia) 2020.

# Script to solve for the instrumental polarization in pol-converted 
# (synthetic) EU-VGOS observations.
# This script complements the contents of the paper:

# I. Marti-Vidal, A. Mus, P. de Vicente & J. Gonzalez (2020), ... TBC




import numpy as np
import pylab as pl
import pickle as pk
import glob
import matplotlib.pyplot as plt






# Name of visibility dataset:
vis = 'EU-VGOS_Simul_1.ms_IF'

# Source properties (and antenna Dterms) used in the simulation:
iff = open('simulation_values_1.dat','rb')
dts,sous=pk.load(iff)
iff.close()

# Figure out some metadata:
nant = len(dts)
nspw = len(dts[0][0])
fnam = [s[0] for s in sous]



# Solve (will take a while!)
if True:
  polsolve(vis=vis,
         field=','.join(fnam), # Solve using all sources
         spw='0~%i'%(nspw-1), # Solve all IFs together
         DRSolve=[True for i in range(nant)], # Solve Dterms for all antennas (R)
         DLSolve=[True for i in range(nant)], # Solve Dterms for all antennas (L)
         CLEAN_models = [[1.0] for i in fnam], # Source models are centered point sources
         frac_pol=[[0.0] for i in fnam], # Initial guess of the fractional polarization
         DR = [0.0j,0.0j,0.0j],DL = [0.2j,0.2j,0.2j], # Initial guesses of the Dterms
         EVPA = [[0.0] for i in fnam], # Initial guesses of the sources' EVPAs
         PolSolve = [[True] for i in fnam], # Solve for the sources polarization as well.
         bound_frac_pol = 0.0, # No bounds for fractional polarization 
#                          (if no boudns, the fit will be done in (Q,U) space.
#                          For this simulated dataset, a fit in (Q,U) works better than in (p,EVPA).
         linear_approx=False, # Solve using polsolve's non-linear Dterm model
         maxiter=2, # Just two iterations of Levenberg-Marquardt are enough.
         do_Faraday = True, # Solve for Faraday rotation (and source (de)polarization)
         nterms=-1) # If nterms is negative, the "multi-IF" mode is activated. Otherwise,
#                     nterms is the order of the Taylor polinomial (in nu space) 
#                     used to fit the Dterm spectra.




# read the results from the fit (ascii file with the full post-fit covariance matrix):
if nspw==1:
  cov = open('%s.spw_0.PolSolve.CovMatrix'%(vis))
else:
  cov = open('%s.spw_0-%i.PolSolve.CovMatrix'%(vis,nspw-1))



# Parse the source polarization quantities fitted by polsolve
# and the Dterms for each IF:

solved = {} ; solvedErr = {}
dtsolved = {} ; dtsolvedErr = {}
lines = cov.readlines()
for lii,line in enumerate(lines):
 temp = line.split()
 if len(temp)>5:
  if temp[2]=='Field':
    nam = temp[4][1:-2]
    if nam not in solved.keys():
      solved[nam] = [0.,0.,0.,0.]
      solvedErr[nam] = [0.,0.,0.,0.]

    if 'Pfrac' in line:
      solved[nam][0] = float(temp[8])
      solvedErr[nam][0] = float(temp[10])
    elif 'EVPA' in line:
      solved[nam][1] = float(temp[8])
      solvedErr[nam][1] = float(temp[10])
    elif 'RM' in line:
      solved[nam][2] = float(temp[8])
      solvedErr[nam][2] = float(temp[10])
    elif 'Pol. Spix' in line:
      solved[nam][3] = float(temp[9])
      solvedErr[nam][3] = float(temp[11])
    elif 'Stokes Q' in line:
      Qt = float(temp[9]) ; Ut = float(lines[lii+1].split()[9])
      QE = float(temp[11]) ; UE = float(lines[lii+1].split()[11])
      QMC = np.random.normal(Qt,QE,1000); UMC = np.random.normal(Ut,UE,1000)
      PE = np.std(np.sqrt(QMC*QMC+UMC*UMC))
      EVE = np.std(90./np.pi*np.arctan2(UMC,QMC))
      solved[nam][0] = (Qt*Qt+Ut*Ut)**0.5
      solved[nam][1] = 0.5*np.arctan2(Ut,Qt)*180./np.pi
      solvedErr[nam][0] = PE
      solvedErr[nam][1] = EVE


  elif temp[2]=='Antenna':
    anam = temp[3].replace(';','')
    if anam not in dtsolved.keys():
      dtsolved[anam] = [np.zeros(nspw,dtype=np.complex128) for i in [0,1]]
      dtsolvedErr[anam] = [np.zeros(nspw,dtype=np.complex128) for i in [0,1]]
    if 'Dterm R' in line:
      if 'Nu-power' in line:
        ifi = int(temp[7])
        if 'real' in line:
          dtsolved[anam][0][ifi] += float(temp[9])
          dtsolvedErr[anam][0][ifi] += float(temp[11])
        elif 'imag' in line:
          dtsolved[anam][0][ifi] += 1.j*float(temp[9])
          dtsolvedErr[anam][0][ifi] += 1.j*float(temp[11])
      else:
        if 'real' in line:
          dtsolved[anam][0][0] += float(temp[8])
          dtsolvedErr[anam][0][0] += float(temp[10])
        elif 'imag' in line:
          dtsolved[anam][0][0] += 1.j*float(temp[8])
          dtsolvedErr[anam][0][0] += 1.j*float(temp[10])

    if 'Dterm L' in line:
      if 'Nu-power' in line:
        ifi = int(temp[7])
        if 'real' in line:
          dtsolved[anam][1][ifi] += float(temp[9])
          dtsolvedErr[anam][1][ifi] += float(temp[11])
        elif 'imag' in line:
          dtsolved[anam][1][ifi] += 1.j*float(temp[9])
          dtsolvedErr[anam][1][ifi] += 1.j*float(temp[11])
      else:
        if 'real' in line:
          dtsolved[anam][1][0] += float(temp[8])
          dtsolvedErr[anam][1][0] += float(temp[10])
        elif 'imag' in line:
          dtsolved[anam][1][0] += 1.j*float(temp[8])
          dtsolvedErr[anam][1][0] += 1.j*float(temp[10])



cov.close()



# Plot results in a nice figure:

cplPfr = [[],[],[]]
cplEV = [[],[],[]]
cplRM = [[],[],[]]
cplspix = [[],[],[]]

for sou in sous:
  Pfr = (sou[2]**2.+sou[3]**2.)**0.5
  EV = 0.5*np.arctan2(sou[3],sou[2])*180./np.pi
  cplPfr[0].append(Pfr)
  cplPfr[1].append(solved[sou[0]][0])
  cplPfr[2].append(solvedErr[sou[0]][0])
  cplEV[0].append(EV)
  cplEV[1].append(solved[sou[0]][1])
  cplEV[2].append(solvedErr[sou[0]][1])
  cplRM[0].append(sou[5])
  cplRM[1].append(solved[sou[0]][2])
  cplRM[2].append(solvedErr[sou[0]][2])
  cplspix[0].append(sou[4])
  cplspix[1].append(solved[sou[0]][3])
  cplspix[2].append(solvedErr[sou[0]][3])

for i in [0,1,2]:
  cplPfr[i] = np.array(cplPfr[i])
  cplEV[i] = np.array(cplEV[i])
  cplRM[i] = np.array(cplRM[i])
  cplspix[i] = np.array(cplspix[i])

  try:
    cplEV[0][cplEV[0]<0.0] += 180.
    cplEV[1][cplEV[1]<0.0] += 180.
    cplEV[0][cplEV[0]>180.0] -= 180.
    cplEV[1][cplEV[1]>180.0] -= 180.
  except:
    pass

  for j in range(len(cplEV[0])):
    if cplEV[0][j]-cplEV[1][j] > 90.:
      cplEV[1][j] += 180.
    elif cplEV[1][j]-cplEV[0][j] > 90.:
      cplEV[0][j] += 180.



fig = pl.figure(figsize=(16,8))
fig.subplots_adjust(left=0.07,bottom=0.1,right=0.97,top=0.96,wspace=0.37,hspace=0.34)

sub1 = fig.add_subplot(241)
sub1.plot(cplPfr[0],cplPfr[1],'ok')
sub1.errorbar(cplPfr[0],cplPfr[1],cplPfr[2],fmt='k',linestyle='none')
sub1.plot(np.array([0.,0.12]),np.array([0.,0.12]),':k')

sub1.set_xlabel(r'p/I True')
sub1.set_ylabel(r'p/I PolSolve')
sub1.set_xlim((0.001,0.11))
sub1.set_ylim((0.001,0.11))


sub2 = fig.add_subplot(242)
sub2.plot(cplEV[0],cplEV[1],'ok')
sub2.errorbar(cplEV[0],cplEV[1],cplEV[2],fmt='k',linestyle='none')
sub2.plot(np.array([0.,180.]),np.array([0.,180.]),':k')

sub2.set_xlabel(r'EVPA True (deg.)')
sub2.set_ylabel(r'EVPA PolSolve (deg.)')
sub2.set_xlim((0.1,179.9))
sub2.set_ylim((0.1,179.9))
sub2.set_xticks([20,60,100,140])
#sub2.set_yticks([20,60,100,140])

sub3 = fig.add_subplot(245)
sub3.plot(cplRM[0]/100.,cplRM[1]/100.,'ok')
sub3.errorbar(cplRM[0]/100.,cplRM[1]/100.,cplRM[2]/100.,fmt='k',linestyle='none')
sub3.plot(np.array([-1.,1.]),np.array([-1.,1.]),':k')

sub3.set_xlabel(r'RM True (10$^2$ rad/m$^2$)')
sub3.set_ylabel(r'RM PolSolve (10$^2$ rad/m$^2$)')
sub3.set_xlim((-1.2,1.2))
sub3.set_ylim((-1.2,1.2))


sub4 = fig.add_subplot(246)
sub4.plot(cplspix[0],cplspix[1],'ok')
sub4.errorbar(cplspix[0],cplspix[1],cplspix[2],fmt='k',linestyle='none')
sub4.plot(np.array([-1.0,1.]),np.array([-1.,1.]),':k')

sub4.set_xlabel(r'Spec. Idx. True')
sub4.set_ylabel(r'Spec. Idx. PolSolve')

sub4.set_ylim((-2.2,2.2))
sub4.set_xlim((-1.1,1.1))


Nus = np.linspace(4.,6.,nspw)

sub5 = fig.add_subplot(322)
sub5.plot(Nus,dtsolved['0'][0].imag,'or')
sub5.errorbar(Nus,dtsolved['0'][0].imag,dtsolvedErr['0'][0].imag,fmt='k',linestyle='none')
sub5.plot(Nus,dtsolved['0'][0].real,'ob')
sub5.errorbar(Nus,dtsolved['0'][0].real,dtsolvedErr['0'][0],fmt='k',linestyle='none')
sub5.plot(Nus,dtsolved['0'][1].imag,'^r')
sub5.errorbar(Nus,dtsolved['0'][1].imag,dtsolvedErr['0'][0].imag,fmt='k',linestyle='none')
sub5.plot(Nus,dtsolved['0'][1].real,'^b')
sub5.errorbar(Nus,dtsolved['0'][1].real,dtsolvedErr['0'][0],fmt='k',linestyle='none')

sub5.plot(Nus,dts[0][0].imag,'-r',label='Imag')
sub5.plot(Nus,dts[0][0].real,'-b',label='Real')
sub5.plot([],[],'ok',label=r'$D_R$')
sub5.plot([],[],'^k',label=r'$D_L$')

sub5.text(5.,0.18,'ONSALA')
sub5.legend(numpoints=1,ncol=4,loc=8,prop={'size':10})
sub5.set_xlabel('Frequency (GHz)')
sub5.set_ylabel('Dterms')

sub5.set_xlim((3.9,6.1))
sub5.set_ylim((-0.25,0.25))


sub6 = fig.add_subplot(324)
sub6.plot(Nus,dtsolved['1'][0].imag,'or')
sub6.errorbar(Nus,dtsolved['1'][0].imag,dtsolvedErr['1'][0].imag,fmt='k',linestyle='none')
sub6.plot(Nus,dtsolved['1'][0].real,'ob')
sub6.errorbar(Nus,dtsolved['1'][0].real,dtsolvedErr['1'][0],fmt='k',linestyle='none')
sub6.plot(Nus,dtsolved['1'][1].imag,'^r')
sub6.errorbar(Nus,dtsolved['1'][1].imag,dtsolvedErr['1'][0].imag,fmt='k',linestyle='none')
sub6.plot(Nus,dtsolved['1'][1].real,'^b')
sub6.errorbar(Nus,dtsolved['1'][1].real,dtsolvedErr['1'][0],fmt='k',linestyle='none')

sub6.plot(Nus,dts[1][0].imag,'-r',label='Imag')
sub6.plot(Nus,dts[1][0].real,'-b',label='Real')
sub6.plot([],[],'ok',label=r'$D_R$')
sub6.plot([],[],'^k',label=r'$D_L$')

sub6.text(5.,0.18,'YEBES')
#sub6.legend(numpoints=1)
sub6.legend(numpoints=1,ncol=4,loc=8,prop={'size':10})
sub6.set_xlabel('Frequency (GHz)')
sub6.set_ylabel('Dterms')

sub6.set_xlim((3.9,6.1))
sub6.set_ylim((-0.25,0.25))



sub7 = fig.add_subplot(326)
sub7.plot(Nus,dtsolved['2'][0].imag,'or')
sub7.errorbar(Nus,dtsolved['2'][0].imag,dtsolvedErr['2'][0].imag,fmt='k',linestyle='none')
sub7.plot(Nus,dtsolved['2'][0].real,'ob')
sub7.errorbar(Nus,dtsolved['2'][0].real,dtsolvedErr['2'][0],fmt='k',linestyle='none')
sub7.plot(Nus,dtsolved['2'][1].imag,'^r')
sub7.errorbar(Nus,dtsolved['2'][1].imag,dtsolvedErr['2'][0].imag,fmt='k',linestyle='none')
sub7.plot(Nus,dtsolved['2'][1].real,'^b')
sub7.errorbar(Nus,dtsolved['2'][1].real,dtsolvedErr['2'][0],fmt='k',linestyle='none')

sub7.plot(Nus,dts[2][0].imag,'-r',label='Imag')
sub7.plot(Nus,dts[2][0].real,'-b',label='Real')
sub7.plot([],[],'ok',label=r'$D_R$')
sub7.plot([],[],'^k',label=r'$D_L$')

sub7.text(5.,0.18,'WETTZELL')
#sub7.legend(numpoints=1)
sub7.legend(numpoints=1,ncol=4,loc=8,prop={'size':10})
sub7.set_xlabel('Frequency (GHz)')
sub7.set_ylabel('Dterms')

sub7.set_xlim((3.9,6.1))
sub7.set_ylim((-0.25,0.25))







pl.savefig('VGOS_SIMUL.pdf')
pl.show()












