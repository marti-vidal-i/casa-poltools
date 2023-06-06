# PolSimulate - A task to simulate simple full-polarization ALMA data.
#
# Copyright (c) Ivan Marti-Vidal - Nordic ARC Node (2016). 
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>,
# or write to the Free Software Foundation, Inc., 
# 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# a. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# b. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
# c. Neither the name of the author nor the names of contributors may 
#    be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import gc
import os, sys
import numpy as np
import scipy.interpolate as spint
import datetime as dt

from casatasks.private import simutil
from casatasks import clearcal, exportuvfits, importuvfits, rmtables, ft
from casatasks import split as CASAsplit

from casatools import ms as mset
from casatools import table #as tb
from casatools import image #as ia
from casatools import coordsys #as cs
from casatools import simulator #as sm
from casatools import measures #as me
from casatools import quanta #as qa

sm = simulator()
me = measures()
ms = mset()
tb = table()
ia = image()
cs = coordsys()
qa = quanta()


__version__ = '2.0b'


__help__ = """ 

Basic simulator of ALMA/J-VLA (and VLBI) full-polarization observations. 
The output should be imaged with CLEAN (with stokes=IQUV) and the polarization 
vectors should be computed with immath (with options poli and pola). 
See the ALMA Polarization CASA Guide for more information.

Please, cite this software as: 

             Marti-Vidal et al. 2021, A&A, 646, 52

PARAMETERS: 

   vis :: Name of output measurement set.

   reuse :: If True, and the measurement set exists, will reuse it (so the antenna names and source coordinates will be reused as well).

   array_configuration :: Array configuration file, where the antenna coordinates and diameters are set (see the files in data/alma/simmos of your CASA path). Default is an ALMA configuration. The name of the array is taken as the part of the name of this file before the first dot (i.e., it is \'alma\' for the default filename).

   elevation_cutoff :: Minimum allowed elevation for the antennas (degrees).

   feed :: Polarization basis for the measurement set. Can be linear (e.g., for ALMA) or circular (e.g., for VLA). Default is linear.

   mounts :: For VLBI observations, this is a list of the antenna mounts (given in the same order as in the array configuration file). A mount type is specified with two characters. Supported mounts: alt-az (\'AZ\'), equatorial (\'EQ\'), X-Y (\'XY\'), Nasmyth Right (\'NR\') and Nasmyth Left (\'NL\'). Default means all antennas are alt-az. 
             It can also be a dictionary, where the keywords are the antenna names (codes) and the values are the mount types (default is alt-az).

   model_Dt_0 :: List of complex numbers (length equal to the number of antennas). If not empty, the first polarizer (i.e., either R or X, depending on the value of \'feed\') will be contamined with a leakage given by these Dterms.

   model_Dt_1 :: Same as model_Dt_0, but for the L (or Y) polarizer.

   LO :: Frequency of the first LO in Hz (this will define the ALMA band of observation). Default is 100 GHz (i.e., ALMA Band 3).

   BBs :: List with the baseband frequency offsets (in Hz). There will be one spectral window per baseband.

   spw_width :: Width of the spectral windows in Hz (the spws will be centered at each BB).

   nchan :: Number of channels in each spw (all spws will have the same number of channels).

   model_image :: List of four images (Stokes I, Q, U, and V, respectively) to be used as observed extended sources. Image cubes are allowed. Default is to NOT simulate extended sources. BEWARE OF THE CURRENT ALMA LIMITATION for extended sources (should fall within the inner 1/3 of the primary beam FWHM).

   I :: List of Stokes I (in Jy) for a set of point source components to simulate. These are added to the \'model_image\' (if it is provided). Default is to add NO point source components. If spectral index is being simulated, the flux densities are referred to the \'LO\' frequency. Example: I = [1.0] for a 1 Jy point source. \n A filename can also be given to define each component. The content of that file will be a list of sub-components in Difmap format (see help).

   Q_frac :: List of FRACTIONAL Stokes Q (i.e., Qfrac = Q/I) for each of the source components defined in the \'I\' vector above. Example: Q = [0.0] for no Q component. If rotation measure is being simulated, these Q values are referred to the \'LO\' frequency.

   U_frac :: List of FRACTIONAL Stokes U (i.e., Ufrac = U/I) for each of the source components defined in the \'I\' vector above. Example: U = [0.0] for no U component. If rotation measure is being simulated, these Q values are referred to the \'LO\' frequency.

   V_frac :: Same as Q_frac and U_frac, but for Stokes V.

   RM :: List of Rotation Measures (RM, in rad/m**2.) for the source components defined above. Example: RM = [0.0] for no RM.

   spec_index :: List of spectral indices for the sources defined above. Default is no source. Example: [0.0] for a component with a flat spectrum.

   RA_offset :: List of right-ascension offsets (in degrees) for the sources defined above. The first source is assumed to be at the phase center, so all sources will be shifted RAoffset[0] arcsec (so that the first source in the list is at the phase center). If one of the \'I\' elements is a file with a list of sub-components, the \'RAoffset\' will be globally applied to all of them.

   Dec_offset :: Same as RAoffset, but for the declinations.

   spectrum_file :: File with user-defined spectra of I, Q, U, and V. See help for details about the file format. This source WILL BE ADDED TO THE PHASE CENTER, together with the source defined in the model_image model and all those given in the I, Q, U, V, lists.

   phase_center :: Coordinates of the observed source (will override the coordinates defined in model_image, if an image is being used). This keyword MUST BE defined.

   incell :: Pixel size of the model_image. If not empty, will override the original value stored in the image. Example: \'0.1arcsec\'. All the Stokes images (I, Q, U, and V) will be set the same way. USE WITH CARE.

   inbright :: Peak intensity of the I Stokes model_image (will override the original value stored in the image). Default is to use the original brightness unit. All the Stokes images (I, Q, U, and V) will be set the same way. Default (i.e., 0.0) means to NOT scale the image birghtness. USE WITH CARE.

   inwidth :: Width of the frequency channels in the model_image. If not empty, will override the original value stored in the image. Example: \'10MHz\'. All the Stokes images (I, Q, U, and V) will be set the same way.

   innu0 :: Frequency of the first image channel (e.g., \'1GHz\'). Default (empty) means to use the value in the image.

   H0 = If a float is given, it is the Hour Angle at the start of the observations, as observed from the array center (in hr). If a string is given, it is the exact start of the observations (UT time) in the format \'2017/01/01/00:00:00\'.

   onsource_time :: Total effective integration time on the source (in hr). Default is 1 hr.

   observe_time :: Total observing time (i.e., including overheads) in hr. Default is 3h, so there will be an observing efficiency of 0.33 if \'onsource_time\' is set to one hour.

   visib_time :: Integration time per visibility. This is a string, where \'s\' stands for \'seconds\'.

   nscan :: Number of scans. Can be provided as a \'listobs\' file (in that case, observe_time, onsource_time and H0 are not used, but taken from the listobs). If a list in \'listobs\' format is not provided then all scans will be set to equal length. If just an integer is given, the scans will be homogeneously distributed across the total observing time. If a list is given, the values will be taken as the starting times of the scans, relative to the duration of the experiment. For instance, if \'observe_time = 6.0\' then \'nscan = [0., 0.5, 0.75]\' will make three scans, one at the start of the observations, the second one 3 hours later and the third one 4.5 hours after the start of the observations.

   apply_parang :: If True, applies the parallactic-angle correction. \n If False, the data polarization will be given \n in the antenna frame (i.e., just as true raw data).

   export_uvf :: If True, exports the measurement into uvfits format (for its use in e.g., AIPS/Difmap).

   corrupt :: Whether to add random noise to the visibilities.

   seed :: Seed of the random number generator in the sm tool.

   Dt_amp :: Will add random Dterms (antenna-wise). Dt_amp is the typical absolute value of the Dterms (real and imag). The actual values will be computed from a random Gaussian distribution.

   Dt_noise :: Will add random channel-dependent contribution to the Dterms. Dt_noise is the typical residual channel noise in the Dterms (real and imag). The actual values for each frequency channel will be those of Dt_amp PLUS a random Gaussian distribution of width Dt_noise. Default is 0.001, similar to the spectral spread of Dterms seen in the SV ALMA polarization data (see the CASA Guide).

   tau0 :: Atmospheric opacity at zenith.

   t_sky :: Sky temperature (in K).

   t_ground :: Ground temperature (in K).

   t_receiver :: Receiver temperature (in K).






  EXAMPLES OF USE:

  1.- Simulate only a point source with constant fractional polarization of 10%,
  an EVPA of zero degrees, no V Stokes, and a steep spectrum:
  
  model_image=[]
  I = [1.0], Q = [0.1], U = [0.], V = [0.], RM = [0.], spec_index = [-1.0]
  spectrum_file = ''


  
  2.- Simulate an extended source, defined by a set of images
  (one image for each Stokes parameter):

  model_image=['I.image', 'Q.image', 'U.image', 'V.image']
  I = [], Q = [], U = [], RM = [], spec_index = []
  spectrum_file=''

  
  
  3.- Simulate the TWO SOURCES of the previous examples
  TOGETHER (i.e., one OVER THE OTHER):

  model_image=['I.image', 'Q.image', 'U.image']
  I = [1.0], Q = [0.1], U = [0.], V = [0.], RM = [0.], spec_index = [-1.0]
  spectrum_file = ''



  4.- Simulate a point source with a user-defined 
  polarization spectrum for I, Q, U, and V:

  model_image=[]
  I = [], Q = [], U = [], V = [], RM = [], spec_index = []
  spectrum_file='my_spectrum.dat'

  The format of my_spectrum is an ASCII file with several rows 
  (one row per frequency channel). The format of each row is:

  FREQUENCY (HZ)    I (Jy)    Q (Jy)    U (Jy)  V (Jy)
  Some example rows (for a Band 6 simulation) could be:

  246.e9   1.0   0.07   0.05   0.00
  247.e9   1.2   0.06   0.05   0.00 
  248.e9   1.3   0.05   0.04   0.00
  ...

  
  The spectrum will be INTERPOLATED to the frequencies of the 
  spectral windows in the measurement set. BEWARE and check that 
  the spw frequencies are properly covered by the spectrum defined 
  in the spectrum file!



  5.- Simulate a source with two compact component of different 
  polarization. The first component will have: I=1, Q=0.03, U=0.02;
  the second component will have I=0.4, Q=0.02, U=-0.02, with an
  offset of 0.5 mas in RA and 0.2 mas in Dec:

  I = [1.00,  0.40]
  Q = [0.03,  0.02]
  U = [0.02, -0.02]
  V = [0.00,  0.00]
  RAoffset =  [0.0, 0.0005]
  Decoffset = [0.0, 0.0002]
  RM = [0.0, 0.0]
  spec_index = [0.0, 0.0]

  The source will be observed with the VLBA at 5GHz, using two spectral
  windows (IFs) of 64 channels and 512MHz each (one centered at 4744 MHz 
  and the other centered at 5256 MHz):

  LO = 5.e9
  BBs = [4.744e9, 5.256e9]
  spw_width = 512.e6
  nchan = 64


  If you want to mimick the exact uv coverage of a real observation
  (or if you want to have a full control over the observing schedule) 
  you can set \'nscan\' to the name of an ascii file, expected to have 
  the time and scan information in \'listobs\' format. 
  For instance, the contents of that file could be:

  ###################################

  Observed from   11-Apr-2017/01:02:30.0   to   11-Apr-2017/08:45:00.0 (UTC)

  11-Apr-2017/01:02:30.0 - 01:08:30.0     
              01:38:26.8 - 01:44:00.0    
              06:51:00.0 - 07:06:00.0    


  ###################################

  and so on.

  Notice that, in this case, the values of \'observe_time\', \'onsource_time\' 
  and \'H0\' will be ignored.




6.- Source components given in Difmap file format:

The format is one sub-component per line, with the format: TYPE FLUX RAOFFSET DECOFFSET. 

Example:   

 0  1.0   0.00   0.00 
 0  0.5   0.01   0.02 
 0  0.25  0.00  -0.02

In this case, three deltas (type 0) with fluxes of 1.0, 0.5 and 0.25 Jy will define this 
source subcomponent. The offsets will be (0.0,0.0) deg. for the first delta, (0.01,0.02) deg. 
for the second, and (0.0,-0.02) deg. for the third one. 

Currently, only deltas (type 0) are supported.






"""




#####################
# UNIT TEST LINES (JUST FOR DEBUGGING & TESTING):
if __name__=='__main__':

# Set random seed (to get the same values as in the paper):
  np.random.seed(42)
# Get random Dterms from a Gaussian distribution:
  DRs = (np.random.normal(0.,0.20,8)+np.random.normal(0.,0.20,8)*1.j)
  DLs = (np.random.normal(0.,0.20,8)+np.random.normal(0.,0.20,8)*1.j)

  H0 = 0.0
  spectrum_file      =  ""
  vis='SgrA_polsimulate.ms'
  model_image = []
  reuse=False # A new Measurement Set will be made from scratch.
  model_Dt_0 = DRs # Dterms in first polarizer (i.e., R)
  model_Dt_1 = DLs # Dterms in second polarizer (i.e., L)
  feed = 'circular' # The polarizers are actually R and L.
  I = [1.0,0.8,0.6,0.4,0.4] # Stokes I of the 5 source components.
  Q_frac = [0.,0.,0.,0.8,0.0] # Fractional Stokes Q.
  U_frac = [0.,0.,0.,0.5,-0.9] # Fractional Stokes U.
  V_frac = [0.,0.,0.,0.,0.] # Fractional V
  RM = [0.,0.,0.,0.,0.]  # Rotation measures.
  spec_index = [0.,0.,0.,0.,0.] # Spectral indices.
## RA offsets of each of the five source components (given in degrees):
  RA_offset = [0.,-40.e-6/3600.,-80.e-6/3600.,-10.e-6/3600.,-30.e-6/3600.]
## Samce for the Dec offsets (all are zero, since the jet is in E-W direction:
  Dec_offset = [0.,0.,0.,0.,0.]
## Phase center of the source core (that of SgrA*)
  phase_center = "J2000 17h45m40.4230 -29d00m28.0400"
  array_configuration    =  "EHT.cfg" # ascii file with the antenna info.
## Antenna mounts (the same order as the antenna config. file is assumed):
  mounts  =  ['AZ', 'NR', 'NR', 'AZ', 'NL', 'NL', 'AZ', 'NL']
  LO = 2.3e+11; BBs = [0.0]; spw_width = 2.e9; nchan = 1 # Freq. config.
  nscan = "TRACK_C.listobs" # Take observing times from this listobs file.
  visib_time = '6s' # VLBI integration time.
  apply_parang=True # Apply parallactic angle to the data.
  corrupt=True
  seed               =  42
  Dt_amp             =  0.0
  Dt_noise           =  0.001
  tau0               =  0.0
  t_sky              =  250.0
  t_ground           =  270.0
  t_receiver         =  50.0
  elevation_cutoff    =  5.0
  export_uvf = True




def polsimulate(vis = '', reuse = False, array_configuration='alma.out04.cfg', elevation_cutoff = 5.0, feed = 'linear',
                mounts = [], model_Dt_0 = [], model_Dt_1 = [], LO=200.e9, BBs = [-7.e9,-5.e9,5.e9,7.e9], spw_width = 2.e9, nchan = 128, 
                model_image=[], I = [], Q_frac = [], U_frac = [], V_frac = [], RM = [], spec_index = [], RA_offset = [], Dec_offset = [], spectrum_file = '',
                phase_center = 'J2000 00h00m00.00 -00d00m00.00', incell='',inbright=0.0, 
                inwidth='', innu0 = '', H0 = -1.5, 
                onsource_time=1.5, observe_time = 3.0,visib_time='6s',nscan = 50, apply_parang = False, export_uvf=True,
                corrupt = True, seed=42, 
                Dt_amp = 0.00, Dt_noise = 0.001, tau0=0.0, t_sky=250.0, t_ground=270.0, t_receiver=50.0):



########
# Uncomment for testing/debugging
#  return True
#if True:
#if __name__=='__main__':
########

  """ 
     
      Program PolSimulate by I. Marti-Vidal (Univ. Valencia).

      Execute polsimulate() to get some help text. 
  """
     


##################################
#### HELPER FUNCTIONS


# Print messages and errors (on screen and in the log):

  def printError(msg,logfile=""):
    printMsg('\n %s \n'%msg,logfile)
  #  casalog.post('PolSimulate: '+msg)
    raise Exception(msg)


  def printMsg(msg,logfile="",dolog=True):
    print(msg)
    if dolog and len(logfile)>0:
       off = open(logfile,'a')
       tstamp = dt.datetime.now().strftime("%Y-%M-%d/%H:%M:%S")
       print('%s: %s'%(tstamp,msg),file=off)
       off.close()






# Return GMST from UT time (from NRAO webpage):
  def GMST(MJD):
    Days = MJD/86400.  
    t = (Days -51544.0)/36525.
    Hh = (Days - np.floor(Days))
    GMsec = 24110.54841 + 8640184.812866*t + 0.093104*t*t - 0.0000062*t*t*t
    return (GMsec/86400. + Hh)*2.*np.pi









# Read the scan times from a listobs file:

  def readListObs(listfile):

    Mon = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Read listobs file:
    iff = open(listfile).readlines()


# Parse the scan information:
    totscans = []
    starttime = []
    stoptime = []
    ObsTimeTot = 0.0
    foundT0 = False

    for line in iff:
    # Get the exact start and end times of the observations:
      if 'Observed from ' in line:
        temp = line.split()
        t0 = temp[2].split('/')
        d0,mo0,y0 = t0[0].split('-')
        h0,m0,s0 = [hdp for hdp in map(float,t0[1].split(':'))]
        simstart = '%i/%02i/%02i %02i:%02i:%02i'%(int(y0),Mon.index(mo0)+1,int(d0),h0,m0,s0)
        ini = dt.date(int(y0),Mon.index(mo0)+1,int(d0))
        t1 = temp[4].split('/')
        d1,mo1,y1 = t1[0].split('-')
        h1,m1,s1 = [hdp for hdp in map(float,t1[1].split(':'))]
        fin = dt.date(int(y1),Mon.index(mo1)+1,int(d1))
        tdelta = ((fin-ini).days)*24. + (h1-h0 + (m1-m0)/60. + (s1-s0)/3600.)
        foundT0 = True
        day = ini

    # Get the exact times of each scan (accounting for day changes):
      temp = line.split()
      if foundT0 and len(temp)>7 and len(temp[0].split(':'))==3 and len(temp[2].split(':'))==3:
        if '/' in temp[0]:
          temp3, temp2 = temp[0].split('/')
          d1, m1, y1 = temp3.split('-')
          day = dt.date(int(y1),Mon.index(m1)+1,int(d1))
          hini = [hdp for hdp in map(float,temp2.split(':'))]
        else:
          hini = [hdp for hdp in map(float,temp[0].split(':'))]

        hfin = [hdp for hdp in map(float,temp[2].split(':'))]

        scandur = (hfin[0]+hfin[1]/60.+hfin[2]/3600.) - (hini[0]+hini[1]/60.+hini[2]/3600.)
        scanini = (hini[0]+hini[1]/60.+hini[2]/3600.) + 24.*((day-ini).days) - (h0 + m0/60. + s0/3600.)
      
      # Add scan to the list:
        #nscan += 1
        starttime.append('%.3fs'%(scanini*3600.))
        stoptime.append('%.3fs'%((scanini+scandur)*3600.))
        ObsTimeTot += scandur

# Return all needed metadata:
  #  print starttime, stoptime, ObsTimeTot, simstart
    return starttime, stoptime, ObsTimeTot, simstart




# END OF HELPER FUNCTIONS
################################################





  printMsg( 'POLSIMULATE - VERSION %s  - University of Valencia'%__version__,dolog=False)





  if len(vis)==0:
     print(__help__)
     return

  LOGNAME = '%s_PolSimulate.log'%os.path.basename(vis)

  util = simutil.simutil()




  array = os.path.basename(array_configuration).split('.')[0].upper()


# ALMA bands:
  Bands = {'3':[84,119],'5':[163,211],'6':[211,275],'7':[275,370],'8':[385,500],'9':[602,720],'10':[787,950]}

# Receiver evector angles. Not used by now:
  Pangs = {'3':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0}


  if array=='ALMA':
    found = False
    for band in Bands.keys():
      if LO/1.e9 > Bands[band][0] and LO/1.e9 < Bands[band][1]:
        found = True
        selb = band
        break

    if not found:
      printError("Frequency %.5fGHz does NOT correspond to any ALMA band!"%(LO/1.e9),LOGNAME)
    else:
      printMsg( 'This is a Band %s ALMA observation.\n'%selb,LOGNAME)

  if feed in ['linear','circular']:
    printMsg('Will simulate feeds in %s polarization basis'%feed,LOGNAME)
  else:
    printError('Unknown feed %s'%feed,LOGNAME)


# Load the different models:

# Point source(s):
  if len(set([len(I),len(Q_frac),len(U_frac),len(V_frac),len(RM),len(spec_index),len(RA_offset),len(Dec_offset)]))>1:
    printError("ERROR! I, Q_frac, U_frac, V_frac, RM, RAoffset, Decoffset and spec_index should all have the same length!",LOGNAME)

  Itemp = [] ; Qtemp = [] ; Utemp = [] ; Vtemp = []; RAtemp = []; Dectemp = [] ; Spectemp = []; RMtemp = []
  Itemp2 = [] ; Qtemp2 = [] ; Utemp2 = [] ; Vtemp2 = []; RAtemp2 = []; Dectemp2 = [] ; Spectemp2 = []; RMtemp2 = []

  for i in range(len(I)):
    if type(I[i]) is str:
      if not os.path.exists(I[i]):
        printError('File %s not found!'%I[i],LOGNAME)
      inpfile = open(I[i])
      for line in inpfile.readlines():
        if not (line.startswith('!') or line.startswith('#')) and len(line.split())==4:
          temp = [hdp for hdp in map(float,line.split())]
          Itemp.append(temp[1]) ; Qtemp.append(temp[1]*Q_frac[i]) ; Utemp.append(temp[1]*U_frac[i]) ; Vtemp.append(temp[1]*V_frac[i])
          RMtemp.append(float(RM[i])) ; RAtemp.append(temp[2]+RA_offset[i]) ; Dectemp.append(temp[3]+Dec_offset[i])
          Spectemp.append(float(spec_index[i]))
      inpfile.close()      
    else:
      Itemp2.append(float(I[i])); Qtemp2.append(float(Q_frac[i]*I[i]))
      Utemp2.append(float(U_frac[i]*I[i])); Vtemp2.append(float(V_frac[i]*I[i]))
      RMtemp2.append(float(RM[i])); Spectemp2.append(float(spec_index[i]))
      RAtemp2.append(float(RA_offset[i])); Dectemp2.append(float(Dec_offset[i]))

  I = Itemp2 + Itemp; Q = Qtemp2 + Qtemp; U = Utemp2 + Utemp; V = Vtemp2 + Vtemp; RM = RMtemp2 + RMtemp
  spec_index = Spectemp2 + Spectemp ; RA_offset = RAtemp2 + RAtemp ; Dec_offset = Dectemp2 + Dectemp

  NmodComp = max([len(I),1])
  

  if len(I)>0:
    printMsg('There are %i delta components. Total flux: %.2e Jy'%(len(I),np.sum(I)),LOGNAME)
  
#  print I,Q,U,V,RM,spec_index,RAoffset,Decoffset


# Point source (user-defined spectrum):
  ismodel = False
  if type(spectrum_file) is str and len(spectrum_file)>0:
    if not os.path.exists(spectrum_file):
      printError("ERROR! spectrum_file is not found!",LOGNAME)
    else:
     try:
      ismodel = True
      iff = open(spectrum_file)
      lines = iff.readlines() ; iff.close()
      model = np.zeros((5,len(lines)))
      for li,line in enumerate(lines):
        temp = [hdp for hdp in map(float,line.split())]
        model[:,li] = temp[:5]
      model2 = model[:,np.argsort(model[0,:])]
      interpI = spint.interp1d(model2[0,:],model2[1,:])
      interpQ = spint.interp1d(model2[0,:],model2[2,:])
      interpU = spint.interp1d(model2[0,:],model2[3,:])
      interpV = spint.interp1d(model2[0,:],model2[4,:])
      printMsg('A spectral-line model source will be simulated',LOGNAME)
     except:
      printError("ERROR! spectrum_file has an incorrect format!",LOGNAME)



# Extended source (and cube!):
  if type(model_image) is list:
    if len(model_image)>0:
      if len(model_image)!= 4:
        printError('ERROR! \'model_image\' list must have four elements (one per Stokes)',LOGNAME)

      new_mod = [m + '.polsim' for m in model_image]

      printMsg('Model images have been provided.',LOGNAME)

# Modify images, if needed:
      FACTOR = 1.0
      for i in range(4):
        if os.path.exists(new_mod[i]):    
           rmtables(new_mod[i])
 
        os.system('cp -r %s %s'%(model_image[i], new_mod[i]))
      #  returnpars = util.modifymodel(model_image[i], new_mod[i],
      #                         '', innu0,incell,
      #                         phase_center,inwidth,0,
      #                         flatimage=False)


        ia.open(new_mod[i])
 #
        mycs = ia.coordsys()
 #
        if float(inbright)>0.0:
 #
          DATA = ia.getchunk()
 #
          if i==0:
            FACTOR = float(inbright)/np.max(DATA)
 #
          DATA *= FACTOR
          ia.putchunk(DATA)
 

        if len(phase_center)>0:

          KK = np.copy(mycs.referencepixel()['numeric'])
          dirs = phase_center.split()
          mycs.setdirection(refcode=dirs[0], refval=' '.join(dirs[1:]))
          mycs.setreferencepixel(type='direction',value = KK[:2])


    #    if len(incell)>0:
    #       newcell = qa.quantity(incell) 
    #       mycs.setincrement(type='direction', value = newcell)


    #    if len(inwidth)>0:
    #       newwidth = qa.quantity(inwidth)
    #       mycs.setincrement(type='spectral',value = newwidth)


    #    if len(innu0)>0:
    #      newnu0 = qa.convertfreq(innu0, mycs.units()[-1])['value']
    #      mycs.setreferencevalue(type='spectral',value=newnu0)

        #   mycs.setreferencevalue(type='stokes',value=1.0)

        ia.setcoordsys(mycs.torecord())

        ia.close()



      Iim, Qim, Uim, Vim = new_mod
    else:
      Iim = '';Qim = '';Uim = ''; Vim = '';
  else:
    printError("ERROR! Unkown model_image!",LOGNAME)



  if len(Iim)>0 and not (os.path.exists(Iim) and os.path.exists(Qim) and os.path.exists(Uim) and os.path.exists(Vim)):
    printError("ERROR! one or more model_image components does not exist!",LOGNAME)


  if len(model_image)==0 and len(I)==0 and not ismodel:
    printError("ERROR! No model specified!",LOGNAME)
  

  if not os.path.exists(array_configuration):
    antlist = os.getenv("CASAPATH").split(' ')[0] + "/data/alma/simmos/"+array_configuration
  else:
    antlist = array_configuration

#  if True:
  try:
    tempUtil = util.readantenna(antlist)
    if len(tempUtil)==7:
      stnx, stny, stnz, stnd, antnames, arrname, arrpos = tempUtil
    else:
      stnx, stny, stnz, stnd, antnames, arrname, arrpos, dummy = tempUtil
    nant = len(antnames)

#  else:
  except:
    printError('ERROR with array configuration file!',LOGNAME)


## If a dictionary is given, generate the list of mounts:
  if type(mounts) is dict:
      auxMounts = ['AZ' for i in range(nant)]
      for anti in mounts.keys():
         if anti not in antnames:
            printError('Antenna %s is not known'%str(anti))
         antidx = antnames.index(anti)
         auxMounts[antidx] = str(mounts[anti])
      mounts = auxMounts


## Check the list of mounts:
  if type(mounts) is not list:
     printError('ERROR: mounts must be a list of strings',LOGNAME)
  elif len(mounts)>0:
      if len(mounts) != nant:
        printError('ERROR: length of \'mounts\' should be %i'%nant,LOGNAME)
      else:
        for mi,mnt in enumerate(mounts):
          if mnt not in ['AZ','XY','EQ','NR','NL']:
              printError('ERROR: Unknown mount type %s'%mnt,LOGNAME)
          else:
              printMsg('   ANTENNA %s WILL HAVE MOUNT %s'%(antnames[mi],mnt),LOGNAME)
  else:
      mounts = ['AZ' for i in range(nant)]
      printMsg('ALL ANTENNAS ARE ASSUMED TO HAVE ALT-AZ MOUNTS',LOGNAME)
      
              

# Setting noise
  if corrupt:
   eta_p, eta_s, eta_b, eta_t, eta_q, t_rx = util.noisetemp(telescope=array,freq='%.9fHz'%(LO))
   eta_a = eta_p * eta_s * eta_b * eta_t
   if t_receiver!=0.0:
    t_rx = abs(t_receiver)
   tau0 = abs(tau0)
   t_sky = abs(t_sky)
   t_ground = abs(t_ground)
  else:
   printMsg('Since \'corrupt\' is False, noise will NOT be added!',LOGNAME)
   Dt_noise = 0.0
   Dt_amp = 0.0


  try:
      model_Dt_0 = list(model_Dt_0)
      model_Dt_1 = list(model_Dt_1)
  except:
      printError('ERROR: model_Dt_0 and model_Dt_1 must be lists of complex numbers!',LOGNAME)


  if len(model_Dt_0) not in [0,nant]:
      printError('ERROR: model_Dt_0 should have %i elements!'%nant,LOGNAME)
  else:
     for it in range(len(model_Dt_0)):
          model_Dt_0[it] = np.complex128(model_Dt_0[it])
             
  if len(model_Dt_1) not in [0,nant]:
      printError('ERROR: model_Dt_1 should have %i elements!'%nant,LOGNAME)
  else:
     for it in range(len(model_Dt_1)):
          model_Dt_1[it] = np.complex128(model_Dt_1[it])

  if len(model_Dt_0)==0:
      model_Dt_0 = [0.+1.j*0. for i in range(nant)]
      
  if len(model_Dt_1)==0:
      model_Dt_1 = [0.+1.j*0. for i in range(nant)]
  




  usehourangle = False
  mount = 'alt-az'
  integ = visib_time

  if float(elevation_cutoff) < 0.0:
    printMsg('WARNING! Negative elevation! Will reset it to zero',LOGNAME)
    minElev = 0.0
  else:    
    minElev = float(elevation_cutoff)*np.pi/180.


  if os.path.exists(vis) and reuse:
    printMsg('\n     WILL REUSE THE INPUT MEASUREMENT SET!  \n',LOGNAME)

  else:  

    os.system('rm -rf '+vis)
    sm.open(vis)


# Setting the observatory and the observation:

    ALMA = me.observatory(array)
    if len(ALMA.keys())==0:
      ALMA = me.observatory('VLBA')

    if type(H0) is str:
      refdate = H0
      H0 = 0.0
    else:
      H0 = float(H0)
      refdate='2017/01/01/00:00:00'
      usehourangle = True


    sm.setconfig(telescopename=array, x=stnx, y=stny, z=stnz,
             dishdiameter=stnd.tolist(),
             mount=mount, antname=antnames, padname=antnames, 
             coordsystem='global', referencelocation=ALMA)

  spwnames = ['spw%i'%i for i in range(len(BBs))]
  dNu = spw_width/nchan/1.e9
  spwFreqs = []
  dtermsX = [] ; dtermsY = []
  ModI = [np.zeros((nchan,NmodComp)) for i in BBs]
  ModQ = [np.zeros((nchan,NmodComp)) for i in BBs]
  ModU = [np.zeros((nchan,NmodComp)) for i in BBs]
  ModV = [np.zeros((nchan,NmodComp)) for i in BBs]


# Spectral windows and D-terms:

  corrp = {'linear':'XX XY YX YY','circular':'RR RL LR LL'}

  for i in range(len(BBs)):
    Nu0 = (LO+BBs[i]-spw_width/2.)/1.e9

    if not reuse:
      sm.setspwindow(spwname=spwnames[i], freq='%.8fGHz'%(Nu0),
               deltafreq='%.9fGHz'%(dNu),
               freqresolution='%.9fGHz'%(dNu), 
               nchannels=nchan, refcode="BARY",
               stokes=corrp[feed])
    spwFreqs.append(1.e9*np.linspace(Nu0,Nu0+dNu*nchan,nchan))
        
  if Dt_amp>0.0:
      DtX = [[np.random.normal(0.,Dt_amp), np.random.normal(0.,Dt_noise)] for j in stnx]
      DtY = [[np.random.normal(0.,Dt_amp), np.random.normal(0.,Dt_noise)] for j in stnx]
  else:
      DtX = [[0.,0.] for j in stnx]
      DtY = [[0.,0.] for j in stnx]

  for i in range(len(BBs)):
    dtermsX.append([np.zeros(nchan,dtype=np.complex128) for j in range(nant)])
    dtermsY.append([np.zeros(nchan,dtype=np.complex128) for j in range(nant)])

  for j in range(nant):
    for i in range(len(BBs)): # TODO: Possibility of setting different Dterms for each spw/channel.

      if type(model_Dt_0[j]) in [list,np.ndarray]:
        dtermsX[i][j][:] = model_Dt_0[j][i]
      else:
        dtermsX[i][j][:] = model_Dt_0[j]

      if type(model_Dt_1[j]) in [list,np.ndarray]:
        dtermsY[i][j][:] = model_Dt_1[j][i]
      else:
        dtermsY[i][j][:] = model_Dt_1[j]

      if Dt_amp > 0.0:
        dtermsX[i][j] += np.random.normal(0.,Dt_amp)*np.exp(1.j*2.*np.pi*np.random.random())
        dtermsY[i][j] += np.random.normal(0.,Dt_amp)*np.exp(1.j*2.*np.pi*np.random.random())
      if Dt_noise > 0.0:
        dtermsX[i][j] += np.random.normal(0.,Dt_noise,nchan)*np.exp(1.j*2.*np.pi*np.random.random(nchan))
        dtermsY[i][j] += np.random.normal(0.,Dt_noise,nchan)*np.exp(1.j*2.*np.pi*np.random.random(nchan))



# Compute point models:

  if len(I)>0:
    for i in range(len(BBs)):
      Lam2 = np.power(299792458./spwFreqs[i],2.)
      LamLO2 = (299792458./spwFreqs[0][0])**2.
      for j in range(len(I)):
        ModI[i][:,j] += I[j]*np.power(spwFreqs[i]/spwFreqs[0][0],spec_index[j])
        p = (Q[j]**2. + U[j]**2.)**0.5*np.power(spwFreqs[i]/spwFreqs[0][0],spec_index[j])
        phi0 = np.arctan2(U[j],Q[j])
        ModQ[i][:,j] += p*np.cos(2.*(RM[j]*(Lam2-LamLO2)) + phi0)
        ModU[i][:,j] += p*np.sin(2.*(RM[j]*(Lam2-LamLO2)) + phi0)
        ModV[i][:,j] += V[j] 
  if ismodel:
    for i in range(len(BBs)):
        ModI[i,0] += interpI(spwFreqs[i])
        ModQ[i,0] += interpQ(spwFreqs[i])
        ModU[i,0] += interpU(spwFreqs[i])
        ModV[i,0] += interpV(spwFreqs[i])


# CASA sm tool FAILS with X Y receiver. Will change it later:
  #  sm.setfeed(mode='perfect R L',pol=[''])
  #  sm.setauto(0.0)


# Field name:
  if len(model_image)>0:

    source = '.'.join(os.path.basename(model_image[0]).split('.')[:-1])
  else:
    source = 'POLSIM'


  isListObs = reuse

  if not reuse:

    sm.setfield(sourcename=source, sourcedirection=phase_center,
          calcode="TARGET", distance='0m')



# Set scans:
    if type(nscan) is str and os.path.exists(nscan):
      isListObs = True
      starttimes, stoptimes, observe_time, refdate = readListObs(nscan)

      scop = len(starttimes)
      sources = [source for i in starttimes]

      printMsg( ' Listobs file being used. \nWill ignore starting time and on-source time from input parameters.\n',LOGNAME)

      mereftime = me.epoch('TAI', refdate)

      sm.settimes(integrationtime=visib_time, usehourangle=False, 
            referencetime=mereftime)


    else:


      mereftime = me.epoch('TAI', refdate)
 
      if usehourangle:
        printMsg( ' Will shift the date of observation to match the Hour Angle range\n',LOGNAME)

      sm.settimes(integrationtime=visib_time, usehourangle=usehourangle, 
            referencetime=mereftime)




      starttimes = []
      stoptimes = []
      sources = []

      try:
        scop = len(nscan)
        scandur = float(onsource_time)/scop
        nscan.sort()
        T0s = [H0 + (float(observe_time)-scandur)/nscan[-1]*sci for sci in nscan]
      except:    
        scandur = float(onsource_time)/nscan
        scop = nscan
        if nscan>1:
          T0s = [H0 + (float(observe_time)-scandur)/(nscan-1)*i for i in range(nscan)]
        else:
          TOs = [H0]


      for i in range(scop):
        sttime = T0s[i]
        endtime = (sttime + scandur)
        if i< scop-1 and endtime > T0s[i+1]:
          printMsg('WARNING! There are overlapping scans! Will shift them ot avoid collisions!',LOGNAME)
          for j in range(i+1,scop):
            T0s[j] += (endtime - T0s[i+1]) + 1.0

        starttimes.append(str(3600.*sttime)+'s')
        stoptimes.append(str(3600.*endtime)+'s')
        sources.append(source)


    for n in range(scop):
      for sp in spwnames: 
        sm.observemany(sourcenames=[sources[n]],spwname=sp,starttimes=[starttimes[n]],stoptimes=[stoptimes[n]],project='polsimulate')
  
    sm.close()






# Change feeds to XY:

  if feed == 'linear':

    printMsg( ' CHANGING FEEDS TO X-Y\n',LOGNAME)
    tb.open(vis+'/FEED',nomodify = False)
    pols = tb.getcol('POLARIZATION_TYPE')
    pols[0][:] = 'X'
    pols[1][:] = 'Y'
    tb.putcol('POLARIZATION_TYPE',pols)
    tb.close()











##############################
# Create an auxiliary MS:
  printMsg( 'Creating the auxiliary single-pol datasets',LOGNAME)
  if feed == 'linear':
    polprods = ['XX','XY','YX','YY','I','Q','U','V']
  elif feed == 'circular':
    polprods = ['RR','RL','LR','LL','I','Q','U','V']

  dvis = [vis +ss for ss in polprods]


  if not reuse:
    for dv in dvis:
      os.system('rm -rf '+dv)

  if (not reuse) or (reuse and (not os.path.exists(dvis[0]))):
      CASAsplit(vis = vis, correlation = polprods[0], datacolumn='data',outputvis=dvis[0])
      clearcal(vis=dvis[0],addmodel=True)
      clearcal(vis=vis,addmodel=True)

#  sm.open(dvis[0])
#  sm.setconfig(telescopename=array, x=stnx, y=stny, z=stnz,
#             dishdiameter=stnd.tolist(),
#             mount=mount, antname=antnames, padname=antnames, 
#             coordsystem='global', referencelocation=ALMA)
#  spwnames = ['spw%i'%i for i in range(len(BBs))]
#  for i in range(len(BBs)):
#    sm.setspwindow(spwname=spwnames[i], freq='%.8fGHz'%((LO+BBs[i]-spw_width/2.)/1.e9),
#               deltafreq='%.9fGHz'%(spw_width/nchan/1.e9),
#               freqresolution='%.9fGHz'%(spw_width/nchan/1.e9), 
#               nchannels=nchan, refcode="BARY",
#               stokes=polprods[0])


#  sm.setfield(sourcename=source, sourcedirection=phase_center,
#          calcode="TARGET", distance='0m')

#  sm.settimes(integrationtime=visib_time, usehourangle=usehourangle, 
#            referencetime=mereftime)

#  for n in range(scop):
#   for sp in spwnames: 
#    sm.observemany(sourcenames=[sources[n]],spwname=sp,starttimes=[starttimes[n]],stoptimes=[stoptimes[n]],project='polsimulate')

#  sm.close()
##############################

  if feed == 'linear':
    printMsg( ' VISIBS WILL BE COMPUTED IN X-Y BASIS',LOGNAME)



# Simulate Stokes parameters:


  for dv in dvis[1:]:
    if (not reuse) or (reuse and (not os.path.exists(dv))):
      os.system('cp -r %s %s'%(dvis[0],dv))


####################################
# Auxiliary arrays (scan-wise):

  # If listobs is used, scans may be of different length.
  # Hence, we load all data at once. Otherwise, we go scan-wise.


  spwscans = []
  for n in range(len(spwnames)):
    ms.open(dvis[0])
    ms.selectinit(datadescid=n)
    if isListObs:
      spwscans.append(np.array([0]))
    else:   
      spwscans.append(np.copy(ms.range('scan_number')['scan_number']))
    ms.close()

  ms.open(dvis[0])
  ms.selectinit(datadescid=0)

  if not isListObs:
    ms.select({'scan_number':int(spwscans[0][0])})

  dataI = np.copy(ms.getdata(['data'])['data']) 
  dataQ = np.copy(dataI) 
  dataU = np.copy(dataI) 
  dataV = np.copy(dataI) 
  ms.close()


  ntimes = np.shape(dataI)[-1]

  if feed=='linear':
    printMsg( ' Simulating X-Y feed observations',LOGNAME)
  else:
    printMsg( ' Simulating R-L feed observations',LOGNAME)


  PAs = []; ant1 = [{} for i in range(len(BBs))] ; ant2 = [{} for i in range(len(BBs))]
  Flags = []

###################################






######################################
# Computing parallactic angles:
  print('\n\n\n')
  printMsg('Computing parallactic angles',LOGNAME)
  
  dirst = phase_center.split()
  csys = cs.newcoordsys(direction=True)
  csys.setdirection(refcode=dirst[0], refval=' '.join(dirst[1:]))
  Dec = csys.torecord()['direction0']['crval'][1]
  RA = csys.torecord()['direction0']['crval'][0]

  CosDec = np.cos(Dec)
  SinDec = np.sin(Dec)

  tb.open(dvis[4]+'/ANTENNA')
  apos = tb.getcol('POSITION')
  Lat = np.arctan2(apos[2,:],np.sqrt(apos[0,:]**2. + apos[1,:]**2.))
  Tlat = np.tan(Lat)
  Lon = np.arctan2(apos[1,:],apos[0,:])
  tb.close()


  for i in range(len(BBs)):
   PAs.append([])
   for ni,n in enumerate(spwscans[i]):

    ms.open(dvis[4],nomodify=False)
    ms.selectinit(datadescid=i)
    if not isListObs:
      ms.select({'scan_number':int(n)})

    temp = ms.getdata(['antenna1','antenna2','u','v','w','time'])
    temp2 = ms.getdata(['flag'])
    temp2['flag'][:] = False
    ant1[i][n] = temp['antenna1']
    ant2[i][n] = temp['antenna2']
    
    Ndata = np.shape(temp['u'])[0]
    PAs[i].append(np.zeros((Ndata,2)))
    
#    V2 = SinDec*temp['v'] - CosDec*temp['w']
    
    
#    Bx = -(apos[0,temp['antenna2']]-apos[0,temp['antenna1']])
#    By = -(apos[1,temp['antenna2']]-apos[1,temp['antenna1']])
#    Bz = -(apos[2,temp['antenna2']]-apos[2,temp['antenna1']])

#    CH = temp['u']*By - V2*Bx
#    SH = temp['u']*Bx + V2*By


    CT1 = CosDec*Tlat[temp['antenna1']]
    CT2 = CosDec*Tlat[temp['antenna2']]
    
#    HAng = np.arctan2(SH,CH)
    HAng = (GMST(temp['time']) - RA)%(2.*np.pi)
    
    H1 = HAng + Lon[temp['antenna1']]
    H2 = HAng + Lon[temp['antenna2']]
    
    
  #  Autos = (CH==0.)*(SH==0.)
    Autos = temp['antenna1']==temp['antenna2']

    H1[Autos] = 0.0
    H2[Autos] = 0.0
    
    E1 = np.arcsin(SinDec*np.sin(Lat[temp['antenna1']])+np.cos(Lat[temp['antenna1']])*CosDec*np.cos(H1))
    E2 = np.arcsin(SinDec*np.sin(Lat[temp['antenna2']])+np.cos(Lat[temp['antenna2']])*CosDec*np.cos(H2))


   # CODE FOR DEBUGGING:
   # if i == 0:
   # 
   #   if ni==0:
   #     OFF = open('TEST_EL.dat','w')
   #     import pickle as pk
   #     pk.dump([temp['antenna1'],temp['antenna2'],temp['time'],180./np.pi*E1,180./np.pi*E2, CH,SH],OFF)
   #     OFF.close()

   # fig = pl.figure()
   # sub = fig.add_subplot(111)
   # APL = 1

   # FLT1 = (temp['antenna1']==APL)*(temp['antenna2']==7)
   # FLT2 = (temp['antenna1']==APL)*(temp['antenna2']==6)

   # sub.plot(temp['time'][FLT1], 180./np.pi*E1[FLT1],'.k')
   # sub.plot(temp['time'][FLT2], 180./np.pi*E1[FLT2],'xr')



    E1[E1<-np.pi] += 2.*np.pi
    E2[E2<-np.pi] += 2.*np.pi
    temp2['flag'][...,np.logical_or(E1<minElev,E1>np.pi)] = True
    temp2['flag'][...,np.logical_or(E2<minElev,E2>np.pi)] = True

 #   print 'GOOD VIS: ',np.sum(np.logical_not(temp2['flag'][0,0,FLT2]))


 #   raw_input('HOLD')


    NscF = np.sum(temp2['flag'][0,0,:])
    if NscF>0:  
      FgFrac = float(NscF)/float(Ndata)*100.  
      if isListObs:
        printMsg('#%i visibs (%.1f%% of data) will be flagged, due to low elevations'%(NscF, FgFrac),LOGNAME)
      else:
        printMsg('#%i visibs (%.1f%% of data) will be flagged, due to low elevations, in scan #%i'%(NscF, FgFrac, n),LOGNAME)


  # Flag autocorrs as well:
    temp2['flag'][...,temp['antenna1']==temp['antenna2']] = True
    

    for j in range(Ndata):
      if mounts[temp['antenna1'][j]] == 'AZ':
          PAs[i][ni][j,0] = -np.arctan2(np.sin(H1[j]), CT1[j] - SinDec*np.cos(H1[j]))
      elif mounts[temp['antenna1'][j]] == 'EQ':
          PAs[i][ni][j,0] = 0.0
      elif mounts[temp['antenna1'][j]] == 'XY':
          PAs[i][ni][j,0] = -np.arctan2(np.cos(H1[j]),SinDec*np.sin(H1[j]))
      elif mounts[temp['antenna1'][j]] == 'NR':
          PAs[i][ni][j,0] = -np.arctan2(np.sin(H1[j]), CT1[j] - SinDec*np.cos(H1[j])) - E1[j]
      elif mounts[temp['antenna1'][j]] == 'NL':
          PAs[i][ni][j,0] = -np.arctan2(np.sin(H1[j]), CT1[j] - SinDec*np.cos(H1[j])) + E1[j]



      if mounts[temp['antenna2'][j]] == 'AZ':
          PAs[i][ni][j,1] = -np.arctan2(np.sin(H2[j]), CT2[j] - SinDec*np.cos(H2[j]))
      elif mounts[temp['antenna2'][j]] == 'EQ':
          PAs[i][ni][j,1] = 0.0
      elif mounts[temp['antenna2'][j]] == 'XY':
          PAs[i][ni][j,1] = -np.arctan2(np.cos(H2[j]),SinDec*np.sin(H2[j]))
      elif mounts[temp['antenna2'][j]] == 'NR':
          PAs[i][ni][j,1] = -np.arctan2(np.sin(H2[j]), CT2[j] - SinDec*np.cos(H2[j])) - E2[j]
      elif mounts[temp['antenna2'][j]] == 'NL':
          PAs[i][ni][j,1] = -np.arctan2(np.sin(H2[j]), CT2[j] - SinDec*np.cos(H2[j])) + E2[j]
    
    if i==0:
      Flags.append(np.copy(temp2['flag'][0,0,:]))      


 #   del E1, E2, H1, H2, HAng, CT1, CT2, CH, SH, V2, Bx, By, Bz
    del E1, E2, H1, H2, HAng, CT1, CT2

    ms.close()  
#######################################










  if len(Iim)>0:
   print("\nSimulating Stokes I")
   for spi in range(len(BBs)):
    ft(vis=dvis[4], spw = str(spi), model = Iim, usescratch=True)


  if len(Qim)>0:
   print("\nSimulating Stokes Q")
   for spi in range(len(BBs)):
    ft(vis=dvis[5], spw = str(spi), model = Qim, usescratch=True)


  if len(Uim)>0:
   print("\nSimulating Stokes U")
   for spi in range(len(BBs)):
    ft(vis=dvis[6], spw = str(spi), model = Uim, usescratch=True)


  if len(Vim)>0:
   print("\nSimulating Stokes V")
   for spi in range(len(BBs)):
    ft(vis=dvis[7], spw = str(spi), model = Vim, usescratch=True)

  print('\n\n')
  printMsg( 'Computing the cross-correlations',LOGNAME)
  XX = np.zeros(np.shape(dataI),dtype=np.complex128)
  YY = np.zeros(np.shape(dataI),dtype=np.complex128)
  XY = np.zeros(np.shape(dataI),dtype=np.complex128)
  YX = np.zeros(np.shape(dataI),dtype=np.complex128)

#  if corrupt:
  XXa = np.zeros(nchan,dtype=np.complex128)
  YYa = np.zeros(nchan,dtype=np.complex128)
  XYa = np.zeros(nchan,dtype=np.complex128)
  YXa = np.zeros(nchan,dtype=np.complex128)
  XXb = np.zeros(nchan,dtype=np.complex128)
  YYb = np.zeros(nchan,dtype=np.complex128)
  XYb = np.zeros(nchan,dtype=np.complex128)
  YXb = np.zeros(nchan,dtype=np.complex128)


  FouFac = 1.j*2.*np.pi*(np.pi/180.)
  for i in range(len(BBs)):
   printMsg( 'Doing spw %i'%i,LOGNAME)
   gc.collect()
   for sci,sc in enumerate(spwscans[i]):
    sys.stdout.write('\r Scan %i of %i    '%(sci+1,len(spwscans[i])))
    sys.stdout.flush()
    ms.open(dvis[4],nomodify=False)
    ms.selectinit(datadescid=i)
    if not isListObs:
      ms.select({'scan_number':int(sc)})

    UVs = ms.getdata(['u','v'])
    U = UVs['u'][np.newaxis,:]/(3.e8/spwFreqs[i][:,np.newaxis])
    V = UVs['v'][np.newaxis,:]/(3.e8/spwFreqs[i][:,np.newaxis])

    if len(Iim)>0:
      dataI[:] = ms.getdata(['model_data'])['model_data']
    else:
      dataI[:] = 0.0
    if len(I)>0 or ismodel:
      for modi in range(NmodComp):
        dataI[:] += ModI[i][:,modi][:,np.newaxis]*np.exp(FouFac*((RA_offset[modi]-RA_offset[0])*U + (Dec_offset[modi]-Dec_offset[0])*V)) 

    ms.close()

    ms.open(dvis[5],nomodify=False)
    ms.selectinit(datadescid=i)
    if not isListObs:
      ms.select({'scan_number':int(sc)})
 
    if len(Qim)>0:
      dataQ[:] = ms.getdata(['model_data'])['model_data']
    else:
      dataQ[:] = 0.0
    if len(I)>0 or ismodel:
      for modi in range(NmodComp):
        dataQ[:] += ModQ[i][:,modi][:,np.newaxis]*np.exp(FouFac*((RA_offset[modi]-RA_offset[0])*U + (Dec_offset[modi]-Dec_offset[0])*V))

    ms.close()

    ms.open(dvis[6],nomodify=False)
    ms.selectinit(datadescid=i)

    if not isListObs:
      ms.select({'scan_number':int(sc)})

    if len(Uim)>0:
      dataU[:] = ms.getdata(['model_data'])['model_data']
    else:
      dataU[:] = 0.0
    if len(I)>0 or ismodel:
      for modi in range(NmodComp):
        dataU[:] += ModU[i][:,modi][:,np.newaxis]*np.exp(FouFac*((RA_offset[modi]-RA_offset[0])*U + (Dec_offset[modi]-Dec_offset[0])*V))

    ms.close()

    ms.open(dvis[7],nomodify=False)
    ms.selectinit(datadescid=i)

    if not isListObs:
      ms.select({'scan_number':int(sc)})

    if len(Vim)>0:
      dataV[:] = ms.getdata(['model_data'])['model_data']
    else:
      dataV[:] = 0.0
    if len(I)>0 or ismodel:
      for modi in range(NmodComp):
        dataV[:] += ModV[i][:,modi][:,np.newaxis]*np.exp(FouFac*((RA_offset[modi]-RA_offset[0])*U + (Dec_offset[modi]-Dec_offset[0])*V))

    ms.close()

    del U, V, UVs['u'], UVs['v']

    for j in range(ntimes):
     PA = PAs[i][sci][j,:]
     C = np.cos(PA) ; S = np.sin(PA)
     EPA = np.exp(1.j*(PA[0]+PA[1])) ; EMA = np.exp(1.j*(PA[0]-PA[1]))


  # Visibilities in the antenna frame:
     if feed == 'linear':
         XXa[:] = ((dataQ[0,:,j]+dataI[0,:,j])*C[0]-(dataU[0,:,j]-1.j*dataV[0,:,j])*S[0])*C[1] - ((dataU[0,:,j]+1.j*dataV[0,:,j])*C[0] + (dataQ[0,:,j]-dataI[0,:,j])*S[0])*S[1]
         XYa[:] = ((dataU[0,:,j]+1.j*dataV[0,:,j])*C[0]+(dataQ[0,:,j]-dataI[0,:,j])*S[0])*C[1] + ((dataQ[0,:,j]+dataI[0,:,j])*C[0] - (dataU[0,:,j]-1.j*dataV[0,:,j])*S[0])*S[1]
         YXa[:] = ((dataU[0,:,j]-1.j*dataV[0,:,j])*C[0]+(dataQ[0,:,j]+dataI[0,:,j])*S[0])*C[1] + ((dataQ[0,:,j]-dataI[0,:,j])*C[0] - (dataU[0,:,j]+1.j*dataV[0,:,j])*S[0])*S[1]
         YYa[:] = -((dataQ[0,:,j]-dataI[0,:,j])*C[0]-(dataU[0,:,j]+1.j*dataV[0,:,j])*S[0])*C[1] + ((dataU[0,:,j]-1.j*dataV[0,:,j])*C[0] + (dataQ[0,:,j]+dataI[0,:,j])*S[0])*S[1]

     if feed == 'circular':
         XXa[:] = (dataI[0,:,j] + dataV[0,:,j])*EMA 
         YYa[:] = (dataI[0,:,j] - dataV[0,:,j])/EMA 
         XYa[:] = (dataQ[0,:,j] + 1.j*dataU[0,:,j])*EPA
         YXa[:] = (dataQ[0,:,j] - 1.j*dataU[0,:,j])/EPA


  # Apply leakage:
     XX[0,:,j] = XXa + YYa*dtermsX[i][ant1[i][sc][j]]*np.conjugate(dtermsX[i][ant2[i][sc][j]]) + XYa*np.conjugate(dtermsX[i][ant2[i][sc][j]]) + YXa*dtermsX[i][ant1[i][sc][j]]
     YY[0,:,j] = YYa + XXa*dtermsY[i][ant1[i][sc][j]]*np.conjugate(dtermsY[i][ant2[i][sc][j]]) + XYa*dtermsY[i][ant1[i][sc][j]] + YXa*np.conjugate(dtermsY[i][ant2[i][sc][j]])
     XY[0,:,j] = XYa + YYa*dtermsX[i][ant1[i][sc][j]] + XXa*np.conjugate(dtermsY[i][ant2[i][sc][j]]) + YXa*dtermsX[i][ant1[i][sc][j]]*np.conjugate(dtermsY[i][ant2[i][sc][j]])
     YX[0,:,j] = YXa + XXa*dtermsY[i][ant1[i][sc][j]] + YYa*np.conjugate(dtermsX[i][ant2[i][sc][j]]) + XYa*dtermsY[i][ant1[i][sc][j]]*np.conjugate(dtermsX[i][ant2[i][sc][j]])


# Put back into sky frame:

     if apply_parang:

       if feed == 'linear':
         XXb[:]= (C[0]*XX[0,:,j] + S[0]*YX[0,:,j])*C[1] + (C[0]*XY[0,:,j]+S[0]*YY[0,:,j])*S[1]
         YYb[:] = -(S[0]*XY[0,:,j] - C[0]*YY[0,:,j])*C[1] + (S[0]*XX[0,:,j]-C[0]*YX[0,:,j])*S[1]
         XYb[:] = (C[0]*XY[0,:,j] + S[0]*YY[0,:,j])*C[1] - (C[0]*XX[0,:,j] + S[0]*YX[0,:,j])*S[1]
         YXb[:] = -(S[0]*XX[0,:,j] - C[0]*YX[0,:,j])*C[1] - (S[0]*XY[0,:,j] - C[0]*YY[0,:,j])*S[1]

         XX[0,:,j] = XXb         
         XY[0,:,j] = XYb
         YX[0,:,j] = YXb
         YY[0,:,j] = YYb

       if feed == 'circular':
         XX[0,:,j] /= EMA
         YY[0,:,j] *= EMA
         XY[0,:,j] /= EPA
         YX[0,:,j] *= EPA


     del PA, C, S, EPA, EMA


  # Save:
    sys.stdout.write(' %s'%polprods[0]) ; sys.stdout.flush()
    ms.open(str(dvis[0]),nomodify=False)
    ms.selectinit(datadescid=i)


    if not isListObs:
      ms.select({'scan_number':int(sc)})
    aux = ms.getdata(['data'])

 #   print 'XX',np.shape(XX)
 #   print i
 #   print np.sum(np.abs(XX))


    aux['data'][:] = XX[:]
    ms.putdata(aux)
    ms.close()
    del aux['data']


    sys.stdout.write(' %s'%polprods[1]) ; sys.stdout.flush()
    ms.open(str(dvis[1]),nomodify=False)
    ms.selectinit(datadescid=i)

    if not isListObs:
      ms.select({'scan_number':int(sc)})
    aux = ms.getdata(['data'])
    aux['data'][:] = XY[:]
    ms.putdata(aux)
    ms.close()
    del aux['data']


    sys.stdout.write(' %s'%polprods[2]) ; sys.stdout.flush()
    ms.open(str(dvis[2]),nomodify=False)
    ms.selectinit(datadescid=i)

    if not isListObs:
      ms.select({'scan_number':int(sc)})
    aux = ms.getdata(['data'])
    aux['data'][:] = YX[:]
    ms.putdata(aux)
    ms.close()
    del aux['data']


    sys.stdout.write(' %s'%polprods[3]) ; sys.stdout.flush()
    ms.open(str(dvis[3]),nomodify=False)
    ms.selectinit(datadescid=i)

    if not isListObs:
      ms.select({'scan_number':int(sc)})
    aux = ms.getdata(['data'])
    aux['data'][:] = YY[:]
    ms.putdata(aux)
    ms.close()
    del aux['data']

  gc.collect()




# The sm tool IS BROKEN for full-polarization datasets!
# Write full-pol MS manually.

  if corrupt:
   print('\n\n\n')
   printMsg( 'Corrupting',LOGNAME)
   for i in range(len(BBs)):
    printMsg( 'Doing spw %i'%i,LOGNAME)
    for pri,pr in enumerate(dvis[:4]):
     print('Polprod %s'%polprods[pri])
     sm.openfromms(pr)
     sm.setseed(seed+4*i + 16*pri)
     sm.setdata(fieldid=[source],spwid=i)
     sm.setnoise(spillefficiency=eta_s,correfficiency=eta_q,
                 antefficiency=eta_a,trx=t_rx,
                 tau=tau0,tatmos=t_sky,tground=t_ground,tcmb=2.725,
                 mode="tsys-manual",senscoeff=-1)
     sm.corrupt()
     sm.done()

# Copy into full-pol ms:
  print('\n\n\n')
  printMsg( 'Saving',LOGNAME)

  for i in range(len(BBs)):
    printMsg( 'Doing spw %i'%i,LOGNAME)
    for pri,pr in enumerate(dvis[:4]):
     print('Polprod %s'%polprods[pri])
     for sc in spwscans[i]:
      ms.open(str(pr),nomodify=False)
      ms.selectinit(datadescid=i)
      if not isListObs:
        ms.select({'scan_number':int(sc)})
      aux = ms.getdata(['data'])['data'][0,:]
      ms.close()
      ms.open(vis,nomodify=False)
      ms.selectinit(datadescid=i)
      if not isListObs:
        ms.select({'scan_number':int(sc)})
      data = ms.getdata(['data'])
      data['data'][pri,:] = aux
      ms.putdata(data)
      ms.close()
      del data['data'], aux



# Flag out auto-correlations and negative elevations:
  printMsg('Flagging autocorrelations and low elevations.',LOGNAME)
  for i in range(len(BBs)):
    ms.open(vis,nomodify=False)
    ms.selectinit(datadescid=i)
    flagsa = ms.getdata(['antenna1','antenna2'])
    flags = ms.getdata(['flag'])
    autos = flagsa['antenna1'] == flagsa['antenna2']
    flags['flag'][:,:,autos] = True
    ms.putdata(flags)
    ms.close()
    del flagsa['antenna1'],flagsa['antenna2'],flags['flag'],autos
    for si,sc in enumerate(spwscans[i]):
      ms.open(vis,nomodify=False)
      ms.selectinit(datadescid=i)
      if not isListObs:
        ms.select({'scan_number':int(sc)})
      flags = ms.getdata(['flag'])
      flags['flag'][:] = False
      flags['flag'][...,Flags[si]] = True
      ms.putdata(flags)
      ms.close()
      del flags['flag']

# Update mounts in ANT table:
  tb.open(vis+'/ANTENNA',nomodify=False)
  MNT = tb.getcol('MOUNT')
  for mti, mnt in enumerate(mounts):
    if mnt == 'EQ':
        MNT[mti] = 'equatorial'
    elif mnt == 'XY':
        MNT[mti] = 'X-Y'
    elif mnt == 'NR':
        MNT[mti] = 'NASMYTH-R'
    elif mnt == 'NL':
        MNT[mti] = 'NASMYTH-L'

  tb.putcol('MOUNT',MNT)
  tb.close()
 



# Re-number scans:

  inttime = float(visib_time[:-1])
  tb.open(vis,nomodify=False)
  TT = tb.getcol('TIME')
  SC = tb.getcol('SCAN_NUMBER')
  TU = np.unique(TT)
  scans = []
  t0 = 0
  for ti in range(1,len(TU)-1):
    if TU[ti+1]-TU[ti] > 2.*inttime:
      scans.append([TU[t0],TU[ti]])
      t0 = ti+1
  scans.append([TU[t0],TU[-1]])

  for si,sc in enumerate(scans):
    SC[np.logical_and(TT>=sc[0],TT<=sc[1])]=si

  tb.putcol('SCAN_NUMBER',SC)
  tb.close()
 


  printMsg( 'Clearing data',LOGNAME)
  del XX, YY, XY, YX, dataI, dataQ, dataU, dataV
  gc.collect()
  clearcal(vis)

  if not reuse:
    for dv in dvis:
      os.system('rm -rf %s'%dv)


  if export_uvf:
    printMsg('Exporting to UVFITS',LOGNAME)
    os.system('rm -rf %s.uvf'%vis)
    exportuvfits(vis=vis,fitsfile='%s.uvf'%vis,datacolumn='data')
    import pyfits as pf
    temp = pf.open('%s.uvf'%vis,mode='update')
    for mti,mt in enumerate(mounts):
        mtidx = {'AZ':0,'EQ':1,'OR':2,'XY':3,'NR':4,'NL':5}[mt]
        temp[2].data['MNTSTA'][mti] = mtidx
    temp.flush()
    temp.close()

  print('\n DONE!\n')




#if __name__=='__main__':
# polsimulate(vis,array_configuration,feed,LO,BBs,spw_width,nchan,model_image,I,Q,U,V,RM,spec_index,
#   spectrum_file,phase_center,incell,inbright,inwidth,H0,onsource_time,observe_time,visib_time,nscan,
#   corrupt, seed, Dt_amp, Dt_noise,tau0,t_sky,t_ground,t_receiver)


