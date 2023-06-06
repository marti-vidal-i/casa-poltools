# PolSolve - A task to solve fot Dterms with extended calibrator sources.
#
# Copyright (c) Ivan Marti-Vidal - Observatorio de Yebes (2018). 
#                                - Universitat de Valencia (2019).
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


# NOTES: Only supports circular feeds.
#        Antenna mounts always have to be given explicitely.
#        2nd order corrections (slower).
#        Parangle for orbital elements is not implemented.

# TODO: 
#       1. Combine IFs and parameterize frequency dependence.
#
#       2. Work directly on FITS-IDI (or uvfits) files.
#
#       3. Write a more complete documentation.
#
#       4. Solve for RM.
#
#       5. Apply the calibration into the corrected column.   DONE!
#       (to overcome mounts limitations in CASA).
#
#       6. Multi-source self-consistent fitting.
#
#

#import casalith
#casaVersion = casalith.version_string()

import gc
import os, sys
import numpy as np
import pylab as pl
import datetime as dt
from scipy.optimize import minimize
import pickle as pk

from casatools import ms as mset
from casatools import table #as tb
from casatools import image #as ia

import PolSolver as PS

ms = mset()
tb = table()
ia = image()


__version__ = '2.0b'

__help__ = """

Leakage solver for circular polarizers and extended polarization calibrators.

Please, cite this software as: 

             Marti-Vidal et al. 2021, A&A, 646, 52

PARAMETERS:

   vis :: Name of input measurement set. The data should already be calibrated (in bandpass and gains).

   spw :: Spectral window(s) to be fitted. Can be an integer, a list of integers, or a CASA-like selection string. Default = all spws are used.

   field :: Field name (or id) to use as calibrator. Follows CASA syntax for several fields.

   mounts :: List of the antenna mounts (must be given in the same order as the ANTENNA table of the Measurement Set). Use this in case the mounts were not properly imported into the ms.
             A mount type is specified with two characters. Supported mounts: alt-az (\'AZ\'),  equatorial (\'EQ\'),  X-Y (\'XY\'),  Nasmyth Right (\'NR\') and  Nasmyth Left (\'NL\'). Default means all antennas are alt-az.
             It can also be a dictionary, where the keywords are the antenna names (codes) and the values are the mount types (default is alt-az).


   feed_rotation :: Rotation of the feed of each antenna with respect to the local horizontal-vertical frame. One value per antenna. Empty list assumes a null feed angle for all antennas.

   DR :: List of complex numbers (length equal to the number of antennas). If not empty, these are the a-priori values of \n the DR leakage terms to use in the fit. The name of a file, with the list pickled, can also be given.

   DL :: List of complex numbers (length equal to the number of antennas). If not empty, these are the a-priori values of the DL leakage terms to use in the fit. The name of a file, with the list pickled, can also be given.

   DRSolve :: List of booleans (length equal to the number of antennas). If not empty, it will tell which DR terms are fitted. The DR[i] term will be fitted if DRSolve[i] is True. Otherwise, DR[i] will be fixed to its a-priori value. Default (i.e., empty list) means to fit all DRs.

   DLSolve :: Just as DRSolve, but for the DL terms.

   isLinear :: List of booleans (length equal to the number of antennas). If an antenna had originally linear-polarization receivers, and PolConvert has been used to get the data into circular polarization basis, then (to a very good approximation) DR = DL. Hence, setting 'isLinear' to True for the polconverted antennas, will force their Dterms to be DR = DL. Default is isLinear=False for all antennas.

   nterms :: Number of Taylor-expansion coefficients to model the frequency dependence of the Dterms. The frequency for the zeroth Taylor coefficient corresponds to that of the minimum channel frequency in the subset of spws to be fitted. Defaults means to fit constant Dterms in frequency. If negative, different Dterms are fitted to each spw (but the source polarization is kept the same for all data).

   maxiter :: Maximum number of iterations allowed in the Levenberg-Marquardt process (only used if \"linear_approx\" is False).

   antenna_weights :: Dictionary of weight factors for each antenna (e.g., antenna_weight["AA"] = 0.1 will weight ALMA down by a factor 10).

   CLEAN_models :: List of CLEAN model files (CCs given in PRTAB format). Each file will correspond to a source component with the same polarization state. If one number is given (instead of a list of filenames), a centered point source (with that flux density) will be used. If more than one field is fitted, CLEAN_models will be a List of Lists (with one list per field). If names of CASA IQU[V] model image(s) are given (i.e., filenames ended with ".model"), these models will be used and FIXED (i.e., frac_pol, EVPA and PolSolve will NOT be used). If equal to "model_column", the model column (for all correlation products) will be used.

   frac_pol :: List of fractional polarizations (one number per source component). frac_pol values must fall between 0 and 1. If more than one field is fitted, this will be a List of Lists (one list per field).

   EVPA :: List of EVPAs in degrees (one number \n per source component). Angles \n are measured from North to East.\n If more than one field is fitted, \n this will be a List of Lists (one list per field).

   PolSolve :: List of booleans (one per source component) that tell which source components are to be fitted in polarization. If PolSolve[i] is True, the fractional polarization and EVPA of the ith source component will be fitted, together with the antenna Dterms. If False, all Stokes parameters of the ith component will be fixed in the fit. Empty list means to fit the polarization of all the source components. If more than one field is fitted, this will be a List of Lists (one list per field).

   bound_frac_pol :: If larger than 0.0, the fitted fractional linear polarization of all sub-components will be bound to that value. This option implies a non-linear fit (so it overrides the value of linear_approx to False).

   do_Faraday :: If True (default is False), rotation measure and Faraday (de)polarization are fitted to each polarization component. This option implies a non-linear fit (so it overrides the value of linear_approx to False).

   unpol_from_data :: If True (default), assume that the difference between the CLEAN I-Stokes model and the parallel-hands data is due to unmodelled unpolarized flux (this is similar to what LPCAL does). If False, the model for the parallel-hand visibilities will be forced to match the CLEAN I-Stokes model. Usually, False should work better in cases of source models with inaccurate amplitudes, whereas True should work better for models with strong dynamic-range limitations.

   parang_corrected :: If True, the data are assumed to be already corrected for parallactic angle. This is usually the case, unless you are working with data generated with polsimulate with no parang correction.


   plot_parang :: If True, plot the time evolution of the antenna feed angles (i.e., parallactic angle plus correction from the antenna mounts).

   min_elev_plot :: In degrees. If plot_parang is True, points with elevations lower than this limit will be plotted in red. THIS DOES NOT FLAG THE DATA. If you want to flag them, run the flagdata task.

   wgt_power :: Power for the visibility weights. Unity means to leave the weights untouched (i.e., equivalent to natural weighting, but for the fit). Zero means equal weights for all visibilities (i.e., equivalent to uniform weighting for the fit).

   rewgt_pfrac :: Modify the weights by 1/(1+rewgt_pfrac*p), where p is the source contribution to the visibility fractional polarization (estimated from the best-fit model with the original weights).

   linear_approx :: If True, solve the polarimetry by assuming linear dependence of visibilities with Dterms (i.e., a la LPCAL; faster).

   plot_residuals :: If True, plot the residual cross-hand visibilities vs. difference of parallactic angle.


POLSOLVE EXAMPLES: 

Let's suppose that we have a CLEAN image in AIPS with two CC tables 
from IMAGR (one per source component).
We produce two ascii files (e.g., CC1.dat and CC2.dat) with the 
output of PRTAB:

  PRTAB(BPRINT=1, EPRINT=0, NDIG = 8, OUTPR = \'PWD:CC1.dat\', 
        DOCRT = -1, INVER = 1, INEXT = \'CC\')

  PRTAB(BPRINT=1, EPRINT=0, NDIG = 8, OUTPR = \'PWD:CC2.dat\', 
        DOCRT = -1, INVER = 2, INEXT = \'CC\')

(if polsolve fails using these files, try to remove all unnecessary 
lines from them, and only keep the model information).

Then, if we want to solve for all Dterms and the Stokes parameters 
of these two components, the keywords to use would be:

  field = \'Name of Calibrator\'
  CLEAN_models = [\'CC1.dat\' , \'CC2.dat\']
  frac_pol = [0., 0.]
  EVPA = [0., 0.]

 -If there are 8 antennas, and one of them (e.g., the fifth one) 
   has a Nasmyth-Left mount:

     mounts = [\'AZ\' for i in range(8)] ; mounts[4] = \'NL\'

  
 -If we want to fix (i.e., to NOT fit) the Dterms for L of 
  the first antenna:

    DLSolve = [True for i in range(8)] ; DLSolve[0] = False

  
 -If we want to fix the fractional polarization of the first 
  component to 10% (with EVPA of 20 deg.):

    frac_pol    = [0.10,   0.0]
    EVPA     = [  20.,  0.0]
    PolSolve = [False, True]


After running, the task will create a Dterms table, with 
the name equal to that of the input measurement set plus 
the suffix \'.spwI.Dterms\' (where \'I\' is the number of 
the spectral window used in the fit).

"""


#####################
# UNIT TEST LINES:
if False and __name__=='__main__':

  vis                =  "SgrA_polsimulate.ms"
  spw                =  []
  field              =  "0"
  mounts             =  ['AZ', 'NR', 'NR', 'AZ', 'NL', 'NL', 'AZ', 'NL']
  feed_rotation      =  []
  DR                 =  []
  DL                 =  []
  DRSolve            =  []
  DLSolve            =  []
  isLinear           =  []
  nterms             =  1
  maxiter            =  10
  antenna_weights    =  {}
  CLEAN_models       =  "SgrA_clean.model"
  frac_pol              =  [0.0]
  EVPA               =  [0.0]
  PolSolve           =  [False]
  bound_frac_pol           =  0.0
  do_Faraday          =  False
  unpol_from_data      =  True
  parang_corrected    =  True
  target_field       =  "POLSIM"
  plot_parang        =  False
  min_elev_plot      =  10.0
  wgt_power          =  1.0
  rewgt_pfrac        =  0.0
  linear_approx      =  True
  plot_residuals     =  False


#
#
##################






def polsolve(vis = '', spw=[], field = '0', mounts = [], feed_rotation = [], DR = [], DL = [],
                DRSolve = [], DLSolve = [], isLinear = [], nterms = 1, maxiter=10, antenna_weights = {}, 
                CLEAN_models = [1.0], frac_pol = [0.0], 
                EVPA = [0.0], PolSolve = [True], bound_frac_pol = 0.0, do_Faraday = False, unpol_from_data = True, 
                parang_corrected = True, target_field = '', plot_parang=False, min_elev_plot=10.0, wgt_power=1.0, 
                rewgt_pfrac = 0.0, linear_approx = False, plot_residuals=False):

  """ 
     
      Program PolSolve by I. Marti-Vidal (Univ. Valencia).

      Execute polsolve() to get some help text. 
  """
     

########
# Uncomment for testing/debugging
#  return True
#if __name__=='__main__' and False:
########


  DEBUG = False


  maxiter = int(maxiter)
  DoFaraday = bool(do_Faraday)
  bound_frac_pol = np.abs(float(bound_frac_pol))
  Dofrac_pol = bound_frac_pol>0.0


  STOKES_CODES = {'I': 1,  'Q': 2,  'U': 3,  'V': 4, 
                 'RR': 5, 'RL': 6, 'LR': 7, 'LL': 8,
                 'XX': 9, 'XY':10, 'YX':11, 'YY':12}

  POL_ORDER = ['RR','RL','LR','LL']
  DATA2READ = ['UVW','ANTENNA1','ANTENNA2','DATA','WEIGHT','FLAG', 'TIME','FIELD_ID']



###########################################
# HELPER FUNCTIONS
###########################################


# Helper functions to print info/errors in terminal + logger:

  def printError(msg,logfile=""):
    printMsg('\n %s \n'%msg,logfile)
  #  casalog.post('PolSolve: '+msg)
    raise Exception(msg)

  def printMsg(msg,logfile="",dolog=True):
    print(msg)
    if dolog and len(logfile)>0:
   #   casalog.post('PolSolve: '+msg)
       off = open(logfile,'a')
       tstamp = dt.datetime.now().strftime("%Y-%M-%d/%H:%M:%S")
       print('%s: %s'%(tstamp,msg),file=off)
       off.close()



# Helper function (return list of spws, given in CASA format):

  def getSpws(spw_str):
    """ Takes a string of spw selection and returns a list 
    with the spw ids. Examples of strings:
    
    '0~3'
    '0,1,2,3~5'
    """

    selSpws = []
    for spw in spw_str.split(','):
      if '~' in spw: # spw range (must be integers)
      #  if True:
        try:  
          s0,s1 = [hdp for hdp in map(int,spw.split('~'))] 
      #  else:
        except:
          printError('WRONG FORMAT FOR SPW SELECTION!')
        if s0<0: # Field ids must exist!
          printError('WRONG RANGE OF SPWs!')  
        selSpws += list(range(s0,s1+1))
      else: # spw id is given (must be an integer).
        try:
          selSpws += [int(spw)]
        except:
          printError('WRONG SPW %s!'%spw)
    return selSpws


# Helper function (return list of target fields, given in CASA format):

  def getFields(vis,field_str):
    """ Takes a string of field range selection and returns a list 
    with the field ids. Examples of strings:
    
    '0~3'
    '0,1,2,3~5'
    'M87,3C279'
    """
      
    tb.open(os.path.join(vis,'FIELD'))  
    NAMES = list(tb.getcol('NAME'))
    
    # Remove annoying spaces:
    fields = field_str.replace(' ','')
    tb.close()
    
    # Field entries separated by commas:
    selFields = []
    for fld in fields.split(','):
      if '~' in fld: # Field range (must be integers)
        try:  
          f0,f1 = [hdp for hdp in map(int,fld.split('~'))] 
        except:
          printError('WRONG FORMAT FOR TARGET FIELDS!')
        if f0<0 or f1>len(NAMES): # Field ids must exist!
          printError('WRONG RANGE OF TARGET FIELDS!')  
        selFields += list(range(f0,f1+1))
      elif fld in NAMES: # Field name is given.
        selFields += [NAMES.index(fld)]
      else: # Field id is given (must be an integer).
        try:
          selFields += [int(fld)]
        except:
          printError('WRONG FIELD %s!'%fld)
    SNAME = [str(NAMES[si]) for si in selFields]      
    return selFields,SNAME





# Return GMST from UT time (from NRAO webpage):
  def GMST(MJD):
    Days = MJD/86400.  
    t = (Days -51544.0)/36525.
    Hh = (Days - np.floor(Days))
    GMsec = 24110.54841 + 8640184.812866*t + 0.093104*t*t - 0.0000062*t*t*t
    return (GMsec/86400. + Hh)*2.*np.pi



# Helper function to compute feed (i.e., mount + parallactic) angles:

  def getParangle(metaDATA, DATA, mounts, feedAngles, doplot=False):
      
    """ Returns feed rotation (mount + parangle) for the visibilities
    corresponding to field (id) given in metaDATA. """
      
      
 #   tb.open(os.path.join(vis,'FIELD'))
 #   scoord = tb.getcol('PHASE_DIR')
 #   tb.close()      

# NOTE: Revise this for multi-source measurement sets!!
 #   RA = scoord[0,0,field]
 #   Dec = scoord[1,0,field]

    CosDec = np.cos(metaDATA['Dec'])
    SinDec = np.sin(metaDATA['Dec'])
    RA = metaDATA['RA']    

# Load antenna info:
    apos = metaDATA['apos']
 #   tb.open(os.path.join(vis,'ANTENNA'))
 #   apos = tb.getcol('POSITION')
    nant = len(apos)
 #   tb.close()

    Lat = np.arctan2(apos[2,:],np.sqrt(apos[0,:]**2. + apos[1,:]**2.))
    Tlat = np.tan(Lat)
    Lon = np.arctan2(apos[1,:],apos[0,:])


# Load data:
  #  ms.open(vis)
  #  ms.selectinit(datadescid=spw)
  #  if scan < 0:
  #    ms.select({'field_id':field})
  #  else:
  #    ms.select({'scan_number':scan})
      
  #  DATA = ms.getdata(['u','v','w','antenna1','antenna2', 'time'])
    Nvis  = len(DATA['antenna1'])
    
  #  ms.close()

# Compute PAs using UV coordinates:

    PAs = np.zeros((Nvis,2),dtype=float)
    
    GOODS = np.ones(Nvis,dtype=bool)

 #   V2 = SinDec*DATA['v'] - CosDec*DATA['w']
    
 #   Bx = -(apos[0,DATA['antenna2']]-apos[0,DATA['antenna1']])
 #   By = -(apos[1,DATA['antenna2']]-apos[1,DATA['antenna1']])
 #   Bz = -(apos[2,DATA['antenna2']]-apos[2,DATA['antenna1']])

 #   CH = DATA['u']*By - V2*Bx
 #   SH = DATA['u']*Bx + V2*By
    
    CT1 = CosDec*Tlat[DATA['antenna1']]
    CT2 = CosDec*Tlat[DATA['antenna2']]
    
 #   HAng2 = np.arctan2(SH,CH)
    HAng = (GMST(DATA['time']) - RA)%(2.*np.pi)






    H1 = HAng + Lon[DATA['antenna1']]
    H2 = HAng + Lon[DATA['antenna2']]
    
    
 #   Autos = (CH==0.)*(SH==0.)
    Autos = DATA['antenna1']==DATA['antenna2']

    H1[Autos] = 0.0
    H2[Autos] = 0.0
    
    GOODS[Autos] = False

    E1 = np.arcsin(SinDec*np.sin(Lat[DATA['antenna1']])+np.cos(Lat[DATA['antenna1']])*CosDec*np.cos(H1))
    E2 = np.arcsin(SinDec*np.sin(Lat[DATA['antenna2']])+np.cos(Lat[DATA['antenna2']])*CosDec*np.cos(H2))

    GOODS[E1<min_elev_plot*np.pi/180.] = False
    GOODS[E2<min_elev_plot*np.pi/180.] = False
    GOODS[E1>np.pi/2.] = False
    GOODS[E2>np.pi/2.] = False


    PAZ1 = np.arctan2(np.sin(H1), CT1 - SinDec*np.cos(H1))
    PAZ2 = np.arctan2(np.sin(H2), CT2 - SinDec*np.cos(H2))


# Add mount rotations:

    for mti,mt in enumerate(mounts):
      filt = DATA['antenna1'] == mti 

      if mt=='AZ':
        PAs[filt,0] = -PAZ1[filt]
      elif mt=='EQ':
        PAs[filt,0] = 0.0
      elif mt=='XY':
        PAs[filt,0] = -np.arctan2(np.cos(H1[filt]),SinDec*np.sin(H1[filt]))
      elif mt=='NR':
        PAs[filt,0] = -PAZ1[filt] - E1[filt]
      elif mt=='NL':
        PAs[filt,0] = -PAZ1[filt] + E1[filt]

      PAs[filt,0] += feedAngles[mti]

      filt = DATA['antenna2'] == mti 
      
      if mt=='AZ':
        PAs[filt,1] = -PAZ2[filt]
      elif mt=='EQ':
        PAs[filt,1] = 0.0
      elif mt=='XY':
        PAs[filt,1] = -np.arctan2(np.cos(H2[filt]),SinDec*np.sin(H2[filt]))
      elif mt=='NR':
        PAs[filt,1] = -PAZ2[filt] - E2[filt]
      elif mt=='NL':
        PAs[filt,1] = -PAZ2[filt] + E2[filt]

      PAs[filt,1] += feedAngles[mti]

# Release memory:
    if doplot:
      TT = np.copy(DATA['time']); A1 = np.copy(DATA['antenna1']); A2 = np.copy(DATA['antenna2'])

#    del DATA['antenna1'], DATA['antenna2'], DATA['u'], DATA['v'], DATA['w'], DATA['time']
#    del H1, H2, E1, E2, PAZ1, PAZ2, Bx, By, Bz, CH, SH, CT1, CT2, V2, filt
    del H1, H2, E1, E2, PAZ1, PAZ2, CT1, CT2, filt

# Finished!
    if doplot:
      return [PAs, TT, A1, A2, GOODS]
    else:
      return PAs






# Helper function to print matrices nicely:
  def printMatrix(a,f="  % .5e"):
    print("Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]")
    rows = a.shape[0]
    cols = a.shape[1]
    for i in range(0,rows):
      msg = ''
      for j in range(0,cols):
         msg += f%a[i,j] 
      print(msg)
    print("")



# Helper function that fills the vector of variables,
# given the vector of fitting parameters:
  def setFitVal(p,NmodComp, nant, nterms, FITSOU, DRSolve, DLSolve, isLinear):
    """ Fills vector of variables given the fitting parameters.
    First, Stokes Q and U for all fittable components. 
    Then, the fittable DR and DL (in real/imag base). 
    
    Fittable source components are given by the list of booleans FITSOU.
    Fittable Dterms are given by the lists of booleans DRSolve and DLSolve.
    NmodComp and nant is the total number (i.e., fixed and fittable) of 
    source components and antennas, respectively.
    """

    if DoFaraday:
      RMF = 4
    else:
      RMF = 2


    i = 0
    for l in range(len(NmodComp)):
      Nprev = int(np.sum(NmodComp[:l]))
      for si in range(NmodComp[l]):
        if DoFaraday:
          if FITSOU[l][si]:
            FitVal[4*(si+Nprev)  ]  = p[i] 
            FitVal[4*(si+Nprev)+1]  = p[i+1] 
            FitVal[4*(si+Nprev)+2]  = p[i+2] 
            FitVal[4*(si+Nprev)+3]  = p[i+3] 
            i += 4
       #     RMF = 4
        else:
          if FITSOU[l][si]:
            FitVal[2*(si+Nprev)  ]  = p[i] 
            FitVal[2*(si+Nprev)+1]  = p[i+1] 
            i += 2
        #    RMF = 2

    Nprev = int(np.sum(NmodComp))


    for ni in range(nant):

      if DRSolve[ni]:
        FitVal[RMF*Nprev+2*ni*nterms  ]  = p[i]
        FitVal[RMF*Nprev+2*ni*nterms+1]  = p[i+1]
        i += 2
        for j in range(1,nterms):
          FitVal[RMF*Nprev+2*ni*nterms+2*j] = p[i]
          FitVal[RMF*Nprev+2*ni*nterms+2*j+1] = p[i+1]
          i += 2

    for ni in range(nant):
      if isLinear[ni]:
          FitVal[(RMF*Nprev+2*nant*nterms)+2*ni*nterms  ]  = FitVal[RMF*Nprev+2*ni*nterms  ]
          FitVal[(RMF*Nprev+2*nant*nterms)+2*ni*nterms+1]  = FitVal[RMF*Nprev+2*ni*nterms+1]
          for j in range(1,nterms):
            FitVal[(RMF*Nprev+2*nant*nterms)+2*ni*nterms+2*j] = FitVal[RMF*Nprev+2*ni*nterms+2*j]
            FitVal[(RMF*Nprev+2*nant*nterms)+2*ni*nterms+2*j+1] = FitVal[RMF*Nprev+2*ni*nterms+2*j+1]
      else:
          if DLSolve[ni]:
            FitVal[(RMF*Nprev+2*nant*nterms)+2*ni*nterms  ]  = p[i]
            FitVal[(RMF*Nprev+2*nant*nterms)+2*ni*nterms+1]  = p[i+1]
            i += 2
            for j in range(1,nterms):
              FitVal[(RMF*Nprev+2*nant*nterms)+2*ni*nterms+2*j] = p[i]
              FitVal[(RMF*Nprev+2*nant*nterms)+2*ni*nterms+2*j+1] = p[i+1]
              i += 2


# Computes the Chi2, given a set of fitting parameter values (p)
# and also computes the Hessian and the residuals vector:
  def getChi2(p, NmodComp, nant, nterms, FITSOU, DRSolve, DLSolve, isLinear, doDeriv = True, doResid = False, doWgt = False):
    """ Returns either the Chi square and Hessian (+ residuals) and
    the optimum model flux scaling. It also computes the Hessian, depending 
    on the value of \"doDeriv\". Same arguments as setFitVal."""
    
    setFitVal(p,NmodComp,nant, nterms, FITSOU, DRSolve, DLSolve, isLinear)

    doHess = bool(doDeriv); doRess = bool(doResid) ; doWgts = bool(doWgt)
    return PS.getHessian(doHess,doRess,doWgts)







###########################################
# END OF HELPER FUNCTIONS
###########################################


  printMsg( '\n\n  POLSOLVE - VERSION %s  - I. Marti-Vidal (Universitat de Valencia, Spain)'%__version__,dolog=False)


# SCRIPT STARTS!
  if len(vis)==0:
     printMsg(__help__,dolog=False)
     printMsg('\n\n    WARNING! Currently, PolSolve only handles circular-feed receivers\n\n',dolog=False)
     return True


  LOGNAME = '%s_PolSolve.log'%os.path.basename(vis)

  if os.path.exists(LOGNAME):
    os.system('rm -rf %s'%LOGNAME)



  GoodMounts = ['AZ','EQ','NR','NL','XY']









### Check if model image(s) is(are) going to be used:
  isMod = False

  if CLEAN_models == 'model_column':
    isMod = True
  elif type(CLEAN_models) is str and CLEAN_models.endswith('.model'):
    isMod = True
    CLEAN_models = [CLEAN_models]
    try:
      ia.open(str(CLEAN_models[0]))
      ia.close()
    except:
      printError('ERROR: Invalid CASA model image!',LOGNAME)
  elif type(CLEAN_models) is list:
    if str(CLEAN_models[0]).endswith('.model'):
      isMod = True    
      try:
        ia.open(str(CLEAN_models[0]))
        ia.close()
      except:
        printError('ERROR: Invalid CASA model image!',LOGNAME)
    for cli in range(1,len(CLEAN_models)):
      if (str(CLEAN_models[cli]).endswith('.model') and not isMod) or (isMod and not str(CLEAN_models[cli]).endswith('.model')):
        printError('ERROR: Either all or none calibrator(s) can be modeled with images',LOGNAME)
        try:
          ia.open(str(CLEAN_models[cli]))
          ia.close()
        except:
          printError('ERROR: Invalid CASA model image!',LOGNAME)


  if isMod:
    DATA2READ += ['MODEL_DATA']









# Currently, we only work with circular feeds:
  doCirc = True


  if (Dofrac_pol or DoFaraday) and bool(linear_approx):
    printMsg('\n INFO: bound_frac_pol or do_Faraday override linear_approximation to FALSE!\n',LOGNAME)
    linear_approx = False


  if Dofrac_pol and np.abs(bound_frac_pol) == 0.0:
    printMsg('\n INFO: Solving in non-linear mode. Will bound frac_pol between 0 and 1.\n',LOGNAME)
    bound_frac_pol = 1.0



# Linear Dterm approximation??
  doOrd1 = bool(linear_approx)




# Sanity checks:
  tb.open(os.path.join(vis,'SPECTRAL_WINDOW'))
  Nspw = len(tb.getcol('REF_FREQUENCY'))
  tb.close()

# Default = all spws.
  if len(spw)==0:
    spw = list(range(Nspw))
  elif type(spw) is int:
    spw = [spw]

  if type(spw) is list:
    spwl = [hdp for hdp in map(int,spw)]
    spwstr = ','.join([hdp for hdp in map(str,spw)])
  else:
    spwl = getSpws(str(spw))
    spwstr = str(spw)

  if not ms.open(vis):
      printError('ERROR with input measurement set!',LOGNAME)
  ms.close()

  for spi in spwl:
    ms.open(vis)
    if not ms.selectinit(datadescid=int(spi)):
      printError('ERROR with spw %i!'%spw,LOGNAME)
    ms.close()









# Load the source model components:

# First, some sanity checks:

# 1.- How many calibrator fields do we have?
  calField,SNAME = getFields(vis,str(field))

  if isMod:
    NCalFields = 0
  else:
    NCalFields = len(calField)


# If model column is used, no subcomponents are needed:




  if isMod and CLEAN_models != 'model_column':

    units = {'rad':1.0, 'deg':np.pi/180., 'arcsec':1./3600.*np.pi/180.}

    printMsg('Will load the model images into the MS',LOGNAME)

    tb.open(os.path.join(vis,'SPECTRAL_WINDOW'))
    Nspw = len(tb.getcol('REF_FREQUENCY'))
    LAMB = np.array([2.99792458e8/tb.getcell('CHAN_FREQ',i) for i in range(Nspw)])

## We assume that all spw have the same number of channels:
    Nchan = len(LAMB[0])
    for li in LAMB:
      if len(li) != Nchan:
        tb.close()
        printError('ERROR: All spws should have the same number of channels',LOGNAME)

    tb.close()
    tb.open(vis,nomodify=False)
    UVW = tb.getcol('UVW')
    MOD = tb.getcol('MODEL_DATA')
    FID = tb.getcol('FIELD_ID')
    SPID = tb.getcol('DATA_DESC_ID')
    tb.close()
    
    MOD[:] = 0.0

    for i,si in enumerate(calField):
      printMsg('Loading model for source %s'%SNAME[i],LOGNAME)

      ia.open(CLEAN_models[i])
      CC = np.average(ia.getchunk(),axis=3)
      summ = ia.summary()
      mask = [np.where(CC[:,:,i]!=0.0) for i in range(4)]
      ia.close()

      Dmask = np.where(FID==si)[0]
      PHASE = np.zeros((len(Dmask),Nchan), dtype=np.complex128)


##  STOKES I:
      print('\n LOADING I')
      for j in range(len(mask[0][0])):
        sys.stdout.write('\r Component %i of %i'%(j+1,len(mask[0][0])))
        sys.stdout.flush()
        RA = (mask[0][0][j]-summ['refpix'][0])*summ['incr'][0]*units[summ['axisunits'][0]]
        Dec = (mask[0][1][j]-summ['refpix'][1])*summ['incr'][1]*units[summ['axisunits'][1]]
        PHASE[:] = np.exp(2.*np.pi*1.j*(UVW[0,np.newaxis,Dmask]*RA + UVW[1,np.newaxis,Dmask]*Dec)/LAMB[SPID[Dmask]])
        MOD[0,:,Dmask] += CC[mask[0][0][j],mask[0][1][j],0]*PHASE[:]
      MOD[3,:,Dmask] = MOD[0,:,Dmask]

## STOKES Q:
      print('\n LOADING Q')
      for j in range(len(mask[1][0])):
        sys.stdout.write('\r Component %i of %i'%(j+1,len(mask[1][0])))
        sys.stdout.flush()
        RA = (mask[1][0][j]-summ['refpix'][0])*summ['incr'][0]*units[summ['axisunits'][0]]
        Dec = (mask[1][1][j]-summ['refpix'][1])*summ['incr'][1]*units[summ['axisunits'][1]]
        PHASE[:] = np.exp(2.*np.pi*1.j*(UVW[0,np.newaxis,Dmask]*RA + UVW[1,np.newaxis,Dmask]*Dec)/LAMB[SPID[Dmask]])
        MOD[1,:,Dmask] += CC[mask[1][0][j],mask[1][1][j],1]*PHASE[:]
      MOD[2,:,Dmask] = MOD[1,:,Dmask]


## STOKES U:
      print('\n LOADING U')
      for j in range(len(mask[2][0])):
        sys.stdout.write('\r Component %i of %i'%(j+1,len(mask[2][0])))
        sys.stdout.flush()
        RA = (mask[2][0][j]-summ['refpix'][0])*summ['incr'][0]*units[summ['axisunits'][0]]
        Dec = (mask[2][1][j]-summ['refpix'][1])*summ['incr'][1]*units[summ['axisunits'][1]]
        PHASE[:] = np.exp(2.*np.pi*1.j*(UVW[0,np.newaxis,Dmask]*RA + UVW[1,np.newaxis,Dmask]*Dec)/LAMB[SPID[Dmask]])
        MOD[1,:,Dmask] += 1.j*CC[mask[2][0][j],mask[2][1][j],2]*PHASE[:]
        MOD[2,:,Dmask] += -1.j*CC[mask[2][0][j],mask[2][1][j],2]*PHASE[:]

      del PHASE, Dmask, CC, mask

    tb.open(vis,nomodify=False)
    MOD2 = tb.getcol('MODEL_DATA')
    MOD2[:] = MOD
    tb.putcol('MODEL_DATA',MOD2)
    tb.close()
    del MOD2






######################################################
# Is the subcomponent information in agreement with
# the number of fields??
#  if NCalFields>1:

  if type(frac_pol) is not list or len(frac_pol)==0:
    frac_pol = [0.0]
    printMsg('\n frac_pol is not correctly set. Will use default',LOGNAME)

  if type(EVPA) is not list or len(EVPA)==0:
    EVPA = [0.0]
    printMsg('\n EVPA is not correctly set. Will use default',LOGNAME)

  if type(PolSolve) is not list or len(PolSolve)==0:
    PolSolve = [False]
    printMsg('\n PolSolve is not correctly set. Will use default',LOGNAME)

#  if type(CLEAN_models) is not list or len(CLEAN_models)==0:
#    CLEAN_models = [1.0]
#    printMsg('\n PolSolve is not correctly set. Will use default',LOGNAME)



  if NCalFields == 1:

    if type(CLEAN_models[0]) is not list:
      CLEAN_models = [CLEAN_models]
    if type(frac_pol[0]) is not list:
      frac_pol = [frac_pol]
    if type(EVPA[0]) is not list:
      EVPA = [EVPA]
    if type(PolSolve[0]) is not list:
      PolSolve = [PolSolve]


  if NCalFields == 0:

    printMsg('\n Model is taken from the MS MODEL column',LOGNAME)
    NmodComp = []
    FITSOU = []
    NFITSOU = 0


  else:

    for inpi in [CLEAN_models,frac_pol,EVPA,PolSolve]:  
      if len(inpi)!=NCalFields:  
        printError("ERROR! Check the length of the field/subcomponent lists!",LOGNAME )

    cCLEAN_models = []
    for ci in CLEAN_models:
      if not (type(ci) is list):
        cCLEAN_models.append([ci])
      else:
        cCLEAN_models.append(ci)

    cfrac_pol = []
    for pi in frac_pol:
      if not (type(pi) is list):
        cfrac_pol.append([pi])
      else:
        cfrac_pol.append(pi)

    cEVPA = []
    for ei in EVPA:
      if not (type(ei) is list):
        cEVPA.append([ei])
      else:
        cEVPA.append(ei)

    cPolSolve = []
    for pi in PolSolve:
      cPolSolve.append(pi)
        


    for i in range(NCalFields):
      if len(set([len(cCLEAN_models[i]),len(cfrac_pol[i]),len(cEVPA[i]),len(cPolSolve[i])])) > 1:
        printError("ERROR! Problem with field id %i: CLEAN_models, frac_pol, EVPA, PolSolve, and all their subcomponents should all have the same length!"%calField[i],LOGNAME)


#####################################################



#  if not (type(CLEAN_models) is list):
#    CLEAN_models = [CLEAN_models]  

#  if len(set([len(CLEAN_models),len(frac_pol),len(EVPA),len(PolSolve)]))>1:
#    printError("ERROR! CLEAN_models, frac_pol, EVPA, and PolSolve should all have the same length!")

# How many components do we have for each field?
    NmodComp = [max([len(cCLEAN_models[i]),1]) for i in range(NCalFields)]

# Flux density and offset (RA, Dec) of the deltas for each component and field:
    DELTAS = [[[] for dd in cCLEAN_models[i]] for i in range(NCalFields)]
    for fi in range(NCalFields):
      for mi in range(NmodComp[fi]):
# Case of AIPS-like CC files:
        if type(cCLEAN_models[fi][mi]) is str:
          if not os.path.exists(cCLEAN_models[fi][mi]):
            printError('ERROR! File %s does not exist!'%cCLEAN_models[fi][mi],LOGNAME)
      
          ifile = open(cCLEAN_models[fi][mi])
          for line in ifile.readlines():
            spl = line.split()
            if len(line)>0 and len(spl)==4:
              try:
                test = int(spl[0])  
                DELTAS[fi][mi].append([hdp for hdp in map(float,line.split()[1:4])])
              except:
                pass
          ifile.close()
          if len(DELTAS[fi][mi])==0:
             printError('ERROR! Problem with CLEAN model file %s'%cCLEAN_models[fi][mi],LOGNAME)


# Case of a centered point source:
        else:
          DELTAS[fi][mi].append([float(cCLEAN_models[fi][mi]),0.,0.])

# Polarization of each component:
    PFRAC = [[hdp for hdp in map(float,cfrac_pol[i])] for i in range(NCalFields)]
    POLANG = [[hdp for hdp in map(float,cEVPA[i])] for i in range(NCalFields)]

# Which components have a fittable polarization?
    FITSOU = [[hdp for hdp in map(bool,cPolSolve[i])] for i in range(NCalFields)]
    NFITSOU = np.sum([np.sum(FITSOU[i]) for i in range(NCalFields)])

# Print for checking:
    printMsg('\n There are %i calibration fields'%NCalFields,LOGNAME)
    TotFluxAll = [0.0 for i in range(NCalFields)]
    for i in range(NCalFields):
      printMsg('\nFor field #%i (%s), there are %i model components'%(calField[i],SNAME[i],NmodComp[i]),LOGNAME)
      TotFluxAll[i] = 0.0
      for sou in range(NmodComp[i]):
        TotFlux = np.sum([di[0] for di in DELTAS[i][sou]])
        printMsg('Comp. %i of field id %i is made of %i deltas (%.3f Jy). A-priori pol: %.2e (%.1f deg.)'%(sou+1, calField[i], len(DELTAS[i][sou]),TotFlux,PFRAC[i][sou], POLANG[i][sou]),LOGNAME)
        TotFluxAll[i] += TotFlux


# Load info about calibrator(s):
#  tb.open(os.path.join(vis,'FIELD'))
#  snam = list(tb.getcol('NAME'))
#  tb.close()
#  if field not in snam:
#      try:
#        fid = int(field)
#      except:
#        printError('ERROR with field %s'%field)
#  else:
#      fid = snam.index(field)




# Load antenna info:
  tb.open(os.path.join(vis,'ANTENNA'))
  anam = list(tb.getcol('NAME'))
  nant = len(anam)
  tb.close()



# Sanity checks for antenna info:
  if nterms < 0:
    FitPerIF = True
    nterms = len(spwl)
  else:
    nterms = int(np.max([1,int(nterms)]))
    FitPerIF = False

  try:

    if type(DR) is str:
      if os.path.exists(DR):
        ifile = open(DR,'rb')
        DR = pk.load(ifile)
        ifile.close()

    if type(DL) is str:
      if os.path.exists(DL):
        ifile = open(DL,'rb')
        DL = pk.load(ifile)
        ifile.close()



    DR = list(DR)
    DL = list(DL)
    DRSolve = list(DRSolve)
    DLSolve = list(DLSolve)

## If a dictionary is given, generate the list of mounts:
    if type(mounts) is dict:
      auxMounts = ['AZ' for i in range(nant)]
      for anti in mounts.keys():
         if anti not in anam:
            printError('Antenna %s is not known'%str(anti))
         antidx = anam.index(anti)
         auxMounts[antidx] = str(mounts[anti])
      mounts = auxMounts


    mounts = list(mounts)

  except:
      printError('ERROR: DR and DL must be lists of complex numbers!\n DRSolve and DLSolve must be lists of booleans.\n mounts must be a list of strings.',LOGNAME)


# Default values for Dterms and mounts:

  DRa = np.zeros((nant,nterms),dtype=np.complex128) #  [0.+1.j*0. for i in range(nant)]
  DLa = np.zeros((nant,nterms),dtype=np.complex128) #  [0.+1.j*0. for i in range(nant)]

  ErrR = np.zeros((nant,nterms)) 
  ErrL = np.zeros((nant,nterms)) 

  if len(isLinear) not in [0,nant]:
      printError('ERROR: isLinear should have %i elements!'%nant,LOGNAME)
  else:
     for it in range(len(isLinear)):
          isLinear[it] = bool(isLinear[it])

  if len(isLinear)==0:
      isLinear = [False for i in range(nant)]


  if len(DR) not in [0,nant]:
      printError('ERROR: DR should have %i elements!'%nant,LOGNAME)
  else:
     for it in range(len(DR)):
       if type(DR[it]) is list:
         try:
           for ni in range(nterms):
             DRa[it,ni] = np.complex128(DR[it][ni])
         except:
           printError('Bad dimensions for DR!',LOGNAME)
       else:
          if FitPerIF:
            DRa[it,:] = np.complex128(DR[it])
          else:
            DRa[it,0] = np.complex128(DR[it])
             
  if len(DL) not in [0,nant]:
      printError('ERROR: DL should have %i elements!'%nant,LOGNAME)
  else:
     for it in range(len(DL)):
       if type(DL[it]) is list:
         try:
           for ni in range(nterms):
             if isLinear[it]:
               DLa[it,ni] = DRa[it,ni]
             else:
               DLa[it,ni] = np.complex128(DL[it][ni])
         except:
           printError('Bad dimensions for DR!',LOGNAME)
       else:
          if isLinear[it]:
            if FitPerIF:
              DLa[it,:] = DRa[it,:]
            else:
              DLa[it,0] = DRa[it,0]
          else:
            if FitPerIF:
              DLa[it,:] = np.complex128(DL[it])
            else:
              DLa[it,0] = np.complex128(DL[it])


  if len(DRSolve) not in [0,nant]:
      printError('ERROR: DRSolve should have %i elements!'%nant,LOGNAME)
  else:
     for it in range(len(DRSolve)):
          DRSolve[it] = bool(DRSolve[it])
             
  if len(DLSolve) not in [0,nant]:
      printError('ERROR: DLSolve should have %i elements!'%nant,LOGNAME)
  else:
     for it in range(len(DLSolve)):
          DLSolve[it] = bool(DLSolve[it])





  if len(mounts) not in [0,nant]:
      printError('ERROR: mounts should have %i elements!'%nant,LOGNAME)
  else:
     for it in range(len(mounts)):
       mounts[it] = str(mounts[it])
       if mounts[it] not in GoodMounts:   
         printError('ERROR: unknown mount %s!'%mounts[it],LOGNAME)
         
#  if len(DR)==0:
#      DRa = np.zeros(nant,nterms,dtype=np.complex128) #  [0.+1.j*0. for i in range(nant)]
      
#  if len(DL)==0:
#      DLa =  np.zeros(nant,nterms,dtype=np.complex128) #[0.+1.j*0. for i in range(nant)]
      
  if len(DRSolve)==0:
      DRSolve = [True for i in range(nant)]
      
  if len(DLSolve)==0:
      DLSolve = [True for i in range(nant)]



  if len(mounts)==0:
      mounts = ['AZ' for i in range(nant)]

## Disconnect DLSolve for linear antennas:
  for ai in range(nant):
     if isLinear[ai]:
        DLSolve[ai] = False



# Feed angles:

  FeedAngles = np.zeros(nant)

  try:
    feed_rotation = list(feed_rotation)

  except:
    printError('ERROR: feed_rotation should be a list of floats!',LOGNAME)
   
   
  if len(feed_rotation) not in [0,nant]:
    printError('ERROR: feed_rotation should have %i elements!'%nant,LOGNAME)

  for it in range(len(feed_rotation)):
    try:  
      FeedAngles[it] = np.pi/180.*float(feed_rotation[it]) 
    except:
      printError('ERROR: bad feed rotation for antenna #%i!'%it,LOGNAME)




    

# Print info for sanity checking:
  printMsg('\nThere are %i antennas'%nant,LOGNAME)
  F = {True: 'FITTABLE',False: 'FIXED   '}
  for ai in range(nant):
    if isLinear[ai]:
      printMsg('Ant %i (%s). Mount %s. Leakage: DR0 = DL0 = (%.3f, %.3f, %s)'%(ai,anam[ai],mounts[ai],DRa[ai,0].real, DRa[ai,0].imag,F[DRSolve[ai]]),LOGNAME)
    else:
      printMsg('Ant %i (%s). Mount %s. Leakage: DR0 (%.3f, %.3f, %s); DL0 (%.3f, %.3f, %s)'%(ai,anam[ai],mounts[ai],DRa[ai,0].real, DRa[ai,0].imag,F[DRSolve[ai]], DLa[ai,0].real, DLa[ai,0].imag, F[DLSolve[ai]]),LOGNAME)


# Get polarization info from MS:
  ms.open(vis)
  polprods = [dd[0] for dd in ms.range(['corr_names'])['corr_names']]
  spwFreqs = np.copy(ms.range('chan_freq')['chan_freq'][:,np.array(spwl)])

  FreqPow = [np.require(np.power(spwFreqs/np.min(spwFreqs)-1.0,float(i)).astype(np.float64),requirements=['C','A']) for i in range(1,nterms)]

  Lambdas = [np.power(2.99792458e8/spwFreqs[:,i],2.) for i in range(len(spwl))]
  ms.close()

# More sanity checks:
  for pi in polprods: # By now, only circular feeds are allowed.
    if pi not in ['RR','RL','LR','LL']:  
      printError("ERROR! Wrong pol. product %s"%pi,LOGNAME)

  if len(polprods) != 4:
    printError('Measurement set is not a valid full-polarization dataset!',LOGNAME)





###################################################################
###################################################################
###################################################
#### IF NOT SOLVING FOR ANY DTERM, WILL JUST APPLY THEM.


  if np.sum(DRSolve)>0 or np.sum(DLSolve)>0:



###################################
# Load data (only for calibration fields):
    allMODELPol = []
    allDATAPol = []; allEMA = []; allEPA = []; allWgt = []
    allAN1 = []; allAN2 = []; allCOMPS = []; allQ = []; allSPW = []
    allParAng1 = []; allParAng2 = []
    WgtCorr = []

    metaDATA = {}

    tb.open(vis)
    fid = tb.getcol('FIELD_ID')
    SPIDNAME = 'SPECTRAL_WINDOW_ID'
    if SPIDNAME not in list(tb.colnames()):
      SPIDNAME = 'DATA_DESC_ID'

    SPWARR = tb.getcol(SPIDNAME)

    spid = np.zeros(np.shape(SPWARR),dtype=bool)
    for spi in spwl:
      spid[:] = np.logical_or(SPWARR==spi,spid)
    tb.close()

    tb.open(os.path.join(vis,'FIELD'))
    scoord = tb.getcol('PHASE_DIR')
    RA = scoord[0,0,:]
    Dec = scoord[1,0,:]
    tb.close()
    tb.open(os.path.join(vis,'ANTENNA'))
    apos = tb.getcol('POSITION')
    metaDATA['apos'] = apos
    tb.close()


    FluxFactor = np.ones(len(calField))
    

    for ci,fi in enumerate(calField):

      fldEntry = np.where((fid==fi)*spid)[0]
      if len(fldEntry)==0:
        printError('There are no valid visibilities for field id %i.'%fi,LOGNAME)

      metaDATA['RA'] = RA[fi]
      metaDATA['Dec'] = Dec[fi]



# Select entries for cal fields:
      tb.open(vis)
      selRows = tb.selectrows(fldEntry)  

# Read spw of each visibility:
      DATA = {'spw':np.require(np.copy(selRows.getcol(SPIDNAME)),requirements=['C','A'])}

# Reset spw numbers to internal order:
      spmask = []
      for spwi,spw in enumerate(spwl):
        spmask.append(DATA['spw']==spw)
      for spwi in range(len(spwl)):
        DATA['spw'][spmask[spwi]] = spwi



# Load the visibilities!
      for inputs in DATA2READ:
        DATA[inputs.lower()] = np.require(np.copy(selRows.getcol(inputs),order='C'),requirements=['C','A'])
      tb.close()

# Metadata and flags:
      Nchan,Nvis = np.shape(DATA['data'])[1:]

      GoodData = np.logical_not(DATA['flag']) 
      GoodWgt =  np.copy(DATA['weight']) 




#################
# Lines for testing (same weight for all corr products)
#  GoodData[:] = np.prod(np.logical_not(DATA['flag']),axis=0)[np.newaxis,:]
#  GoodWgt[:] = np.sum(DATA['weight'],axis=0)[np.newaxis,:]
#################


# Weight (set to zero for bad data and autocorrs):
      Wgt = [np.zeros((Nvis,Nchan)) for i in range(4)]
      WgtC = np.zeros(Nvis)

      for i in range(4):
        j = polprods.index(POL_ORDER[i])  
        Wgt[i][:,:] = GoodWgt[j,:,np.newaxis]
        Wgt[i][np.logical_not(np.transpose(GoodData[j,...]))] = 0.0
        Wgt[i][DATA['antenna1']==DATA['antenna2'],:] = 0.0
        Wgt[i][Wgt[i]<0.0] = 0.0

        for aw in antenna_weights.keys():
          if aw in anam:
            if i==0:
               printMsg('Will re-weight %s by a factor %.2e'%(aw,antenna_weights[aw]),LOGNAME)
            Wgt[i][DATA['antenna1']==anam.index(aw),:] *= antenna_weights[aw]
            Wgt[i][DATA['antenna2']==anam.index(aw),:] *= antenna_weights[aw]

# Correct weight power:
        Bads = Wgt[i] == 0.0
        Wgt[i][:] = np.power(Wgt[i],wgt_power)
        Wgt[i][Bads] = 0.0
    


# Arrange data optimally (i.e., [time,channel,polariz]):
      DATAPol = np.zeros((Nvis,Nchan,4),dtype=np.complex128)
      for i in range(4):
        DATAPol[:,:,i] = np.transpose(DATA['data'][polprods.index(POL_ORDER[i]),:,:])
  
      if NCalFields==0:
        MODELPol = np.zeros((Nvis,Nchan,4),dtype=np.complex128)
        for i in range(4):
          MODELPol[:,:,i] = np.transpose(DATA['model_data'][polprods.index(POL_ORDER[i]),:,:])


######################################
# Computing parallactic angles:

      printMsg('\nComputing parallactic (feed) angles for field id %i (%s)'%(fi,SNAME[ci]),LOGNAME)


      if plot_parang:

        NMAX = 8  # Maximum number of antennas in the plot
        col = ['r','g','b','c','m','y','k','w'] # colors

  #    PAs, TT, A1, A2, FG = getParangle(vis,spw,fi,mounts,FeedAngles,doplot=True)
        PAs, TT, A1, A2, FG = getParangle(metaDATA,DATA,mounts,FeedAngles,doplot=True)

        MIN = PAs[:,0] < -np.pi
        PAs[MIN,0] += 2.*np.pi
        MIN = PAs[:,1] < -np.pi
        PAs[MIN,1] += 2.*np.pi

        MAX = PAs[:,0] > np.pi
        PAs[MAX,0] -= 2.*np.pi
        MAX = PAs[:,1] > np.pi
        PAs[MAX,1] -= 2.*np.pi


        UT = TT/86400.
        UT -= int(np.min(UT))
        UT *= 24.

  
        tb.open(vis+'/ANTENNA')
        ANTNAMES = tb.getcol('NAME')
        tb.close()


        OFF = open('polsolve_feed-angles_Field%i.dat'%fi,'w')


# Subsets of plots of NMAX antennas:
        SUBSET = []
        j = 0
        i = 0
        while i< len(ANTNAMES):
          if i+NMAX <= len(ANTNAMES):
            SUBSET.append(ANTNAMES[i:i+NMAX])
            i += NMAX
            j += 1
          else:
            SUBSET.append(ANTNAMES[i:])
            j += 1
            break


        for k in range(j):
          fig = pl.figure()
          sub = fig.add_subplot(111)



          for i in range(len(SUBSET[k])):

            sub.set_ylabel('Feed angle (deg.)')

            if i + NMAX*k >0:  

              mask = (A2==i+NMAX*k)*(A1!=i+NMAX*k)*FG
              TOPLOT = np.copy(-PAs[mask,1]*180./np.pi)
              PLOTTED = np.sum(mask)>0


              sub.plot( UT[mask], TOPLOT, 'o%s'%col[i], label=ANTNAMES[i+NMAX*k])
         
              mask = (A1==i+NMAX*k)*(A2!=i+NMAX*k)*FG
              TOPLOT = np.copy(-PAs[mask,0]*180./np.pi)

              sub.plot( UT[mask], TOPLOT, 'o%s'%col[i])



            else:
              mask = (A1==0)*(A2!=1)*(A2!=0)*FG  #+(A2==3))
              TOPLOT = np.copy(-PAs[mask,0]*180./np.pi)

              sub.plot( UT[mask], TOPLOT, 'o%s'%col[i], label=ANTNAMES[i+NMAX*k])


 
            pl.legend(numpoints=1)
        
          sub.set_xlim((np.min(UT),np.max(UT) + 0.3*(np.max(UT)-np.min(UT))))
          sub.set_ylim((-189.,189.))
          sub.set_xlabel('UT (h)')  

          pl.savefig('%s_FeedAngle_plot_%i_Field%i.png'%(vis,k,fi))
     
        OFF.close()
    
  
      else:
 
        PAs = getParangle(metaDATA, DATA, mounts, FeedAngles)


#######################################
  



################
# Compute Fourier transform of each source component:

      U = (DATA['uvw'][0,:])[:,np.newaxis]/(3.e8/np.transpose(spwFreqs[:,DATA['spw']]))
      V = (DATA['uvw'][1,:])[:,np.newaxis]/(3.e8/np.transpose(spwFreqs[:,DATA['spw']]))
      Q = np.sqrt(np.power(DATA['uvw'][0,:],2.)+np.power(DATA['uvw'][1,:],2.))

      if NCalFields>0:

        COMPS = np.zeros((Nvis,Nchan,NmodComp[ci]+1),dtype=np.complex128)

        FouFac = 1.j*2.*np.pi*(np.pi/180.)

        for i in range(NmodComp[ci]):
          printMsg( 'Computing source component %i of %i'%(i+1,NmodComp[ci]),LOGNAME)
          gc.collect()
    
          for delt in DELTAS[ci][i]:
            COMPS[:,:,i] += delt[0]*np.exp(FouFac*(delt[1]*U + delt[2]*V)) 

          COMPS[:,:,NmodComp[ci]] += COMPS[:,:,i]

# How much flux do we have in the DATA?
        DSUM = np.sum(np.abs(np.average(Wgt[0]*DATAPol[:,:,polprods.index('RR')] + Wgt[3]*DATAPol[:,:,polprods.index('LL')],axis=1)))/(np.sum(np.average(Wgt[0]+Wgt[3],axis=1)))

# How much flux do we have in the MODEL?
        MSUM = np.sum(np.abs(np.average((Wgt[0]+Wgt[3])*COMPS[:,:,NmodComp[ci]],axis=1)))/(np.sum(np.average((Wgt[0]+Wgt[3]),axis=1)))
  
 
# Notice that the Chi2 will take out the effects of this FluxFactor!  
        FluxFactor[ci] = MSUM/DSUM
        printMsg('All model components of field id %i describe up to %.2f percent of the signal.'%(fi,FluxFactor[ci]*100.),LOGNAME)

        if unpol_from_data:
          printMsg('unCLEANed signal will be taken as unpolarized.\n',LOGNAME)
          WgtSum = Wgt[0]+Wgt[3]
          WgtSum[WgtSum==0.0] = 1.0
          COMPS[:,:,NmodComp[ci]] = (Wgt[0]*DATAPol[:,:,polprods.index('RR')] + Wgt[3]*DATAPol[:,:,polprods.index('LL')])/WgtSum
          FluxFactor[ci] = 1.0

####################
##### CODE FOR TESTING (WRITE MODEL INTO MS):
#  ms.open(vis,nomodify=False)
#  ms.selectinit(datadescid=int(spw))
#  ms.select({'field_id':fid})
#  MDATA = ms.getdata(['data','corrected_data','model_data'])
#  MDATA['model_data'][0,0,:] = COMPS[:,0,NmodComp]
#  MDATA['model_data'][1,0,:] = COMPS[:,0,NmodComp]
#  MDATA['model_data'][2,0,:] = COMPS[:,0,NmodComp]
#  MDATA['model_data'][3,0,:] = COMPS[:,0,NmodComp]
#
#  MDATA['corrected_data'][:,:,:] = MDATA['data'] - MDATA['model_data']
#  ms.putdata(MDATA)
#  ms.close()
#  del MDATA['model_data'],MDATA['data'],MDATA['corrected_data']
#  del MDATA
####################




# Sum and difference of PANGS (in complex form):
      EPA = np.exp(1.j*(PAs[:,0]+PAs[:,1])) ; EMA = np.exp(1.j*(PAs[:,0]-PAs[:,1]))

# Get data back into antenna frame (if it was in the sky frame):
      if parang_corrected:
        printMsg("Undoing parang correction",LOGNAME)
        for i in range(4):
          if POL_ORDER[i][0]=='R':
            DATAPol[:,:,i] *= np.exp(1.j*(PAs[:,0]))[:,np.newaxis]
            if NCalFields==0:
              MODELPol[:,:,i] *= np.exp(1.j*(PAs[:,0]))[:,np.newaxis]
          else:
            DATAPol[:,:,i] /= np.exp(1.j*(PAs[:,0]))[:,np.newaxis]
            if NCalFields==0:
              MODELPol[:,:,i] /= np.exp(1.j*(PAs[:,0]))[:,np.newaxis]


          if POL_ORDER[i][1]=='R':
            DATAPol[:,:,i] /= np.exp(1.j*(PAs[:,1]))[:,np.newaxis]
            if NCalFields==0:
              MODELPol[:,:,i] /= np.exp(1.j*(PAs[:,1]))[:,np.newaxis]
          else:
            DATAPol[:,:,i] *= np.exp(1.j*(PAs[:,1]))[:,np.newaxis]
            if NCalFields==0:
              MODELPol[:,:,i] *= np.exp(1.j*(PAs[:,1]))[:,np.newaxis]


      if NCalFields>0:
        allCOMPS.append(np.require(np.copy(COMPS,order='C'),requirements=['C','A']))
      else:
        allMODELPol.append(np.require(np.copy(MODELPol,order='C'),requirements=['C','A']))

      allQ.append(np.require(np.copy(Q,order='C'),requirements=['C','A']))
      allDATAPol.append(np.require(np.copy(DATAPol,order='C'),requirements=['C','A']))
      allEMA.append(np.require(np.copy(EMA,order='C'),requirements=['C','A']))
      allEPA.append(np.require(np.copy(EPA,order='C'),requirements=['C','A']))
      allParAng1.append(np.require(np.copy(180./np.pi*PAs[:,0],order='C'),requirements=['C','A']))
      allParAng2.append(np.require(np.copy(180./np.pi*PAs[:,1],order='C'),requirements=['C','A']))
      allWgt.append([np.require(np.copy(Wgt[i],order='C'),requirements=['C','A']) for i in range(4)])
      allAN1.append(np.require(np.array(DATA['antenna1'],order='C',dtype=np.int32),requirements=['C','A']))
      allAN2.append(np.require(np.array(DATA['antenna2'],order='C',dtype=np.int32),requirements=['C','A']))
      allSPW.append(np.require(np.array(DATA['spw'],order='C',dtype=np.int32),requirements=['C','A']))
      WgtCorr.append(np.require(np.copy(WgtC,order='C'),requirements=['C','A']))


# Free some memory:
      if not DEBUG:
        del GoodData, GoodWgt 
        for i in range(3,-1,-1):
          del Wgt[i]
        del Wgt, WgtC
        del EPA, EMA, DATAPol,Q, U, V
        if NCalFields>0:
          del COMPS    
        for inputs in DATA2READ:
          del DATA[inputs.lower()]
        del DATA 
        gc.collect()



####################################################################
###############################################
########################




#  del DATA['uvw'], DATA['antenna1'], DATA['antenna2'], DATA['time']




# Code for testing (turned off):
#  if False:
#    import pickle as pk
#    OFF = open('FTMODEL.dat','w')
#    pk.dump([U,V,COMPS,DATAPol,Wgt],OFF)
#    OFF.close()
#
################




# Number of fittable parameters (i.e., sources + antennas):   
    Npar = int(2*nterms*(np.sum(DRSolve) + np.sum(DLSolve)) + 2*NFITSOU)

# Number of model variables (i.e., fittable + fixed parameters):
    Nvar = int(4*nant*nterms + 2*int(np.sum(NmodComp)))
    VarSou = NFITSOU # Number of variables related to source components.

## Add Faraday rotation:
    if DoFaraday:
      Nvar += 2*int(np.sum(NmodComp))
      Npar += 2*NFITSOU


# Vectors to store parameter indices in the covariance matrix:
    PAntL = np.require(np.zeros(nant,dtype=np.int32),requirements=['C','A'])
    PAntR = np.require(np.zeros(nant,dtype=np.int32),requirements=['C','A'])
    PSou = np.require(np.zeros(int(np.sum(NmodComp)),dtype=np.int32),requirements=['C','A'])

# All model variables (both fittable and fixed):
    VAntL = np.require(np.zeros(nant,dtype=np.int32),requirements=['C','A'])
    VAntR = np.require(np.zeros(nant,dtype=np.int32),requirements=['C','A'])
    VSou = np.require(np.zeros(int(np.sum(NmodComp)),dtype=np.int32),requirements=['C','A'])

# Are the antennas linear??
#    linAnt = np.require(np.zeros(nant,dtype=np.int32),requirements=['C','A'])
#    for li in range(nant):
#       if isLinear[li]:
#          linAnt[li]=1


# Figure out the indices of each parameter of the cov. matrix:
# First elements are the source Q and U.
# Then, DRs (Re and Im). Then, DLs (Re and Im):
    k = 0
    NPrev = 0
    for l in range(NCalFields):
      NPrev = int(np.sum(NmodComp[:l]))
      for i in range(NmodComp[l]):
        if FITSOU[l][i]:
          PSou[i + NPrev] = k
          k += 2
          if DoFaraday:
            k += 2
        else:
          PSou[i + NPrev] = -1


    for i in range(nant):
      if DRSolve[i]:
        PAntR[i] = k
        k += 2*nterms
      else:
        PAntR[i] = -1


    for i in range(nant):
      if DLSolve[i] and (not isLinear[i]):
        PAntL[i] = k
        k += 2*nterms
      else:
        PAntL[i] = -1


# Initial parameter values:
    InitVal = np.zeros(Nvar)
    i = 0
    for l in range(NCalFields):
      NPrev = int(np.sum(NmodComp[:l]))
      for fi in range(NmodComp[l]):
        if Dofrac_pol:
          InitVal[i] = np.abs(PFRAC[l][fi]) ; InitVal[i+1] = POLANG[l][fi]*np.pi/180.
          if InitVal[i] >= bound_frac_pol or InitVal[i] == 0.0:
            InitVal[i] = bound_frac_pol*0.5
          InitVal[i] = np.arcsin(2.*InitVal[i]/bound_frac_pol-1.)
   #       InitVal[i] = PFRAC[l][fi]*np.cos(POLANG[l][fi]*np.pi/90.)
   #       InitVal[i+1] = PFRAC[l][fi]*np.sin(POLANG[l][fi]*np.pi/90.)
        else:
          InitVal[i] = PFRAC[l][fi]*np.cos(POLANG[l][fi]*np.pi/90.)
          InitVal[i+1] = PFRAC[l][fi]*np.sin(POLANG[l][fi]*np.pi/90.)

        VSou[fi + NPrev] = i
        i += 2
        if DoFaraday:
          i += 2

    for ni in range(nant):
      InitVal[i] = DRa[ni,0].real
      InitVal[i+1] = DRa[ni,0].imag
      for ji in range(1,nterms):
        InitVal[i+2 + 2*(ji-1)] = DRa[ni,ji].real
        InitVal[i+3 + 2*(ji-1)] = DRa[ni,ji].imag
      VAntR[ni] = i
      i += 2*nterms
    for ni in range(nant):
      InitVal[i] = DLa[ni,0].real
      InitVal[i+1] = DLa[ni,0].imag
      for ji in range(1,nterms):
        InitVal[i+2 + 2*(ji-1)] = DLa[ni,ji].real
        InitVal[i+3 + 2*(ji-1)] = DLa[ni,ji].imag
      VAntL[ni] = i
      i += 2*nterms


# Model variables (fittable and fixed):
    FitVal = np.require(InitVal,requirements=['C','A'])

# Memory location for Hessian and residuals vector:
    Hessian = np.require(np.zeros((Npar,Npar),dtype=float),requirements=['C','A'])
    ResVec = np.require(np.zeros(Npar,dtype=float),requirements=['C','A'])

    MaxRes = []

    if plot_residuals:
      printMsg('Plotting Data',LOGNAME)
      Qmax = np.max([np.max(itQ) for itQ in allQ])/1.e6
      fig = pl.figure()
      sub = fig.add_subplot(111)

      tb.open(vis+'/ANTENNA')
      ANTNAMES = tb.getcol('NAME')
      tb.close()

    #MaxRes = np.max(np.abs(allDATAPol))
      foundData = False; plotted = False
      for i in range(len(ANTNAMES)):
        sub.cla()
        MaxRes.append(0.0)
        for s in range(len(allDATAPol)):
          mask1 = np.logical_or(allAN1[s][:]==i,allAN2[s][:]==i)
          for j in range(Nchan):
            mask = np.logical_and(mask1,allWgt[s][POL_ORDER.index('RL')][:,j]>0.0)
            ResVis1 = allDATAPol[s][mask,j,POL_ORDER.index('RL')]
            mask[:] = np.logical_and(mask1,allWgt[s][POL_ORDER.index('LR')][:,j]>0.0)
            ResVis2 = allDATAPol[s][mask,j,POL_ORDER.index('LR')]
            Qplot = allQ[s][mask]/1.e6
            if len(ResVis1)>0:
              foundData = True
              MaxRes[i] = np.max([MaxRes[i],np.max(np.abs(ResVis1))])
              ss = sub.scatter(ResVis1.real,ResVis1.imag,vmin=0.,vmax=Qmax,c=Qplot,edgecolors='none',marker='.')
            if len(ResVis2)>0:
              foundData = True
              MaxRes[i] = np.max([MaxRes[i],np.max(np.abs(ResVis2))])
              ss = sub.scatter(ResVis2.real,ResVis2.imag,vmin=0.,vmax=Qmax,c=Qplot,edgecolors='none',marker='.')
            del ResVis1, ResVis2

        if foundData: 
          if not plotted:
            try:
              plotted = True
              cb = pl.colorbar(ss) 
              cb.set_label(r'Baseline Length ($10^3$ km)')
            except:
              printMsg('WARNING: Problem with colorbar!',LOGNAME)

          sub.set_title('RL & LR DATA (Baselines to %s)'%ANTNAMES[i])
          sub.set_xlabel(r'Real $V_{obs}$ (Jy)')
          sub.set_ylabel(r'Imag $V_{obs}$ (Jy)')
          if MaxRes[i]>0.0:
            sub.set_xlim((-MaxRes[i],MaxRes[i]))
            sub.set_ylim((-MaxRes[i],MaxRes[i]))
          sub.plot(np.array([0.,0.]),np.array([-MaxRes[i],MaxRes[i]]),':k')
          sub.plot(np.array([-MaxRes[i],MaxRes[i]]),np.array([0.,0.]),':k')

          pl.savefig('%s_PolSolve_Data_%s.png'%('.'.join(vis.split('.')[:-1]),ANTNAMES[i]))



# Set memory and send data to C++:

    success = PS.setData(allDATAPol,allWgt, WgtCorr, FreqPow, Lambdas, allSPW, allAN1,allAN2,allCOMPS,allMODELPol,allEPA,allEMA, PSou,PAntR,PAntL,VSou,VAntR,VAntL,FitVal,Hessian,ResVec,FluxFactor,doCirc,doOrd1,FitPerIF,len(spwl),allParAng1,allParAng2,bound_frac_pol,DoFaraday)



    if success != 0:
      printError('ERROR READING DATA INTO C++: CODE %i'%success,LOGNAME)



# Just code for testing (turned off):
#  if False:
#    myfit = minimize(getChi2,[0. for i in range(Npar)], args= (NmodComp, nant, FITSOU, DRSolve, DLSolve), method='nelder-mead')
#    iff = open('polsolve.dterms','w')
#    import pickle as pk
#    pk.dump(myfit,iff)
#    iff.close()
#    print myfit



  




##################################
################################################
##############################################################
####################################################################
# Levenberg-Marquardt in-house implementation:





    def find_step_dogleg(pu, pb, delta):
        return max(np.roots([np.dot(pb-pu,pb-pu), 2*np.dot(pb-pu,pu), np.dot(pu,pu)-delta**2]))


    def dogleg(grad, hess, delta, inverted = False):

        if inverted == False:
            pb = -np.matmul(np.linalg.inv(hess), grad)
        else:
            pb = -np.matmul(hess, grad)
       # Newton step
        if np.linalg.norm(pb) <= delta: 
            return pb
        #print "AFTER NEWTON"
        denom_pu = (np.matmul(np.atleast_1d(np.matmul(grad.T, hess)), grad))
        num_pu = np.matmul(grad.T, grad).astype('f')
        pu = - (num_pu/denom_pu)*grad
        
        # Steppest descent
        if np.linalg.norm(pu) >= delta: 
            return (delta/np.linalg.norm(pu).astype('f'))*pu

        # find optimal curve
        tau = find_step_dogleg(pu, pb, delta)
        return pu + tau*(pb-pu)


    def LMFit(pini):
        NITER = np.min([maxiter,3*Npar]); Lambda = 1.e-6; temp = .5; kfac = 2.5; functol = 1.e-8; change =np.inf

        

    # For testing:
        if DEBUG:
          NITER = 1
          Lambda = 0.0
    ##############

        Chi2 = 0

        p = np.array(pini)
        backupP = np.copy(p)

        bestP = np.array(pini)


        HessianDiag = np.zeros(np.shape(Hessian),dtype=np.float64)
        backupHess = np.zeros(np.shape(Hessian),dtype=np.float64)
        backupGrad = np.zeros(np.shape(ResVec),dtype=np.float64)
        Inverse = np.zeros(np.shape(Hessian),dtype=np.float64)

        Hessian[:,:] = 0.0
        ResVec[:] = 0.0

        CurrChi = getChi2(pini, NmodComp, nant, nterms, FITSOU, DRSolve, DLSolve, isLinear)[0]
        bestChi = float(CurrChi)

        backupHess[:,:] = Hessian
        backupGrad[:] = ResVec



        if DEBUG:
      #    print(pini,NmodComp,nant,nterms,FITSOU,DRSolve,DLSolve)
      #    print(CurrChi)
          print('\n\n  Hessian: ')
          test = printMatrix(Hessian) 
          print('Res. vector: ',ResVec)
      #    raise Exception("STOP!")
        controlIter = 1

        try:
          Inverse[:] = np.linalg.pinv(Hessian)
        except:
          Inverse[:] = np.linalg.inv(Hessian)

        Dpar = np.dot(Inverse,ResVec)
        p += Dpar
        bestP[:] = p
        backupP[:] = p

        if rewgt_pfrac>0.0:
          Chi2,BestRatio = getChi2(p, NmodComp, nant, nterms, FITSOU, DRSolve, DLSolve, isLinear, doDeriv=False, doWgt=True)

          for s in range(NCalFields):
            print('DoWgt from %.3e to %.3e'%(np.min(WgtCorr[s]),np.max(WgtCorr[s])))
            for i in range(4):
              allWgt[s][i] *= 1./(1. + rewgt_pfrac*WgtCorr[s][:,np.newaxis])


        if linear_approx:
          Chi2,BestRatio = getChi2(bestP, NmodComp, nant, nterms, FITSOU, DRSolve, DLSolve, isLinear, doDeriv=False, doResid=True)
          try:
            return [bestP[:], Inverse, Chi2, BestRatio]
          except:
            return False
        else:
          Hessian[:,:] = 0.0
          ResVec[:] = 0.0
          CurrChi = getChi2(bestP, NmodComp, nant, nterms, FITSOU, DRSolve, DLSolve, isLinear)[0]
          if CurrChi<bestChi:
            backupHess[:,:] = Hessian
            backupGrad[:] = ResVec
            try:
              Inverse[:] = np.linalg.pinv(Hessian)
            except:
              Inverse[:] = np.linalg.inv(Hessian)
#
            Dpar = np.dot(Inverse,ResVec)
            p += Dpar
            bestChi = float(CurrChi)
            bestP[:] = p
            backupP[:] = p

        #for i in range(NITER):
        while controlIter <= NITER and change > functol:
          controlIter += 1
          Hessian[:,:] = 0.0
          ResVec[:] = 0.0
          CurrChi = getChi2(p, NmodComp, nant, nterms, FITSOU, DRSolve, DLSolve, isLinear)[0]
          for n in range(len(p)):
            HessianDiag[n,n] = Hessian[n,n]
          try:
            goodsol = True
            Inverse[:] = np.linalg.pinv(Hessian+Lambda*HessianDiag)
            Dpar = np.dot(Inverse,ResVec)
            DirDer = 0.5*np.matmul(np.atleast_1d(np.matmul(Dpar.T, Hessian)), Dpar)
            DirDer2 = np.matmul(ResVec.T, Dpar)
            """
            DirDer = 0.5*sum(Hessian*Dpar.T*Dpar.T)
            DirDer2 = np.dot(ResVec*Dpar, Dpar)
            # TheorImpr = DirDer-2.*DirDer2 
            """
            TheorImpr =  DirDer2+DirDer 
          except:
            goodsol=False
            Dpar = 0.0
            TheorImpr = -10.0

          p += Dpar
          
          h = dogleg(ResVec, Inverse, temp, inverted=True)
          if controlIter==NITER:
            break
          if goodsol:
            Hessian[:,:] = 0.0
            ResVec[:] = 0.0
            Chi2,BestRatio = getChi2(p, NmodComp, nant, nterms,FITSOU, DRSolve, DLSolve, isLinear)
            if Chi2<bestChi:
              bestChi = float(Chi2)
              backupP[:] = p
           # else:
           #   p[:] = backupP

            RealImpr = Chi2 - CurrChi
          else:
            RealImpr = 1.0


          if TheorImpr != 0.0:
            Ratio = RealImpr/(np.matmul(ResVec.T, h)+0.5*np.matmul(np.atleast_1d(np.matmul(h.T, Hessian)), h))
          else:
            Ratio = 0.0

          if Ratio < 0.25:
       #     print "\nIF 1\n" #, Lambda
            temp = .25 * temp
            #if RealImpr<0.0:
            #   temp = np.sqrt(kfac)
            #else:
            #   temp = kfac
          elif Ratio > 0.75 and np.isclose(np.linalg.norm(h), temp, 1e-40):
            temp = np.min([2*temp, kfac])

          elif not goodsol:
            temp = kfac

       #   if Ratio > 1./16:
       #     #temp = chi2
       #     p = p + h
       #     change = np.linalg.norm(ResVec)
       #     #print "CHANGE--->  ",change
#
       #   else:
       #    # temp = 1.0
       #     change = np.inf
            
          Lambda *= temp
          
          if Chi2 == 0.0 or CurrChi==0.0:
            break

          todivide = {True:1.0, False:np.abs(CurrChi)}[CurrChi<functol]
          relchi = np.abs(Chi2-CurrChi)/todivide

          todivide = np.copy(backupP); todivide[np.abs(todivide)<functol] = 1.0
          relpar = np.max(np.abs((p-backupP)/todivide))

          if relchi < functol and relpar < functol: 
          #  p[:] = backupP
            break

          if CurrChi<Chi2:
            p[:] = backupP
          #  Hessian[:,:] = backupHess
          #  ResVec[:] = backupGrad
          else:
            CurrChi = Chi2
            backupHess[:,:] = Hessian
            backupGrad[:] = ResVec
            backupP[:] = p

     #   if controlIter == NITER:
     #     sys.stdout.write("\n\n REACHED MAXIMUM NUMBER OF ITERATIONS!\nThe algorithm may not have converged!\nPlease, check if the parameter values are meaningful.\n")
     #     sys.stdout.flush()
     
   #     if DEBUG:
   #       print 'Hessian\n', Hessian

        try:
          Chi2,BestRatio = getChi2(backupP, NmodComp, nant, nterms,FITSOU, DRSolve, DLSolve, isLinear, doDeriv=False,doResid=True)
          return [backupP[:], np.linalg.pinv(backupHess), Chi2, BestRatio] #getChi2(p, NmodComp, nant, FITSOU, DRSolve, DLSolve, doDeriv=False)]
        except:
          return False




























####################################################################
##############################################################
################################################
##################################











# Prepare vector of initial parameters:
# First elements are Q and U of the fittable source components.
# Then, the fittable Dterms (Re and Im) for R.
# Then, the fittable Dterms (Re and Im) for L.
    i = 0
    pini = np.zeros(Npar)
    for l in range(NCalFields):
      Nprev = np.sum(NmodComp[:l])
      for si in range(NmodComp[l]):
        if FITSOU[l][si]:
          if Dofrac_pol:
            pini[i] = np.abs(PFRAC[l][si]) ; pini[i+1] = POLANG[l][si]*np.pi/180.
            if pini[i]>=bound_frac_pol or pini[i]==0.0:
              pini[i] = bound_frac_pol*0.5
            pini[i] = np.arcsin(2.*pini[i]/bound_frac_pol-1.)
          else:
            Q = PFRAC[l][si]*np.cos(POLANG[l][si]*np.pi/90.)
            U = PFRAC[l][si]*np.sin(POLANG[l][si]*np.pi/90.)
            pini[i] = Q ; pini[i+1] = U

          i += 2
          if DoFaraday:
            i += 2

    for si in range(nant):
      if DRSolve[si]:
        pini[i] = DRa[si,0].real ; pini[i+1] = DRa[si,0].imag
        for ji in range(1,nterms):
          pini[i+2 + 2*(ji-1)] = DRa[si,ji].real
          pini[i+3 + 2*(ji-1)] = DRa[si,ji].imag
        i += 2*nterms

    for si in range(nant):
      if DLSolve[si] and (not isLinear[si]):
        pini[i] = DLa[si,0].real ; pini[i+1] = DLa[si,0].imag
        for ji in range(1,nterms):
          pini[i+2 + 2*(ji-1)] = DLa[si,ji].real
          pini[i+3 + 2*(ji-1)] = DLa[si,ji].imag
        i += 2*nterms


# FIT!

    LM = LMFit(pini)


    if LM == False:
      printError('AN ERROR OCCURRED DURING THE MINIMIZATION PROCESS!',LOGNAME)

    Errors = [np.sqrt(np.abs(LM[1][i,i])*LM[2]) for i in range(Npar)]

#  ErrR = []
#  ErrL = []


  ## Plot residuals:
#### allDATAPol,allWgt,allAN1,allAN2,allCOMPS,allEPA,allEMA
    if plot_residuals:
      printMsg('Plotting residuals',LOGNAME)
      Qmax = np.max([np.max(itQ) for itQ in allQ])/1.e6
      fig = pl.figure()
      sub = fig.add_subplot(111)

      tb.open(vis+'/ANTENNA')
      ANTNAMES = tb.getcol('NAME')
      tb.close()

      foundData = False ; plotted = False
      for i in range(len(ANTNAMES)):
        sub.cla()
        for s in range(len(allDATAPol)):
          mask1 = np.logical_or(allAN1[s][:]==i,allAN2[s][:]==i)
          for j in range(Nchan):
            mask = np.logical_and(mask1,allWgt[s][POL_ORDER.index('RL')][:,j]>0.0)    
            ResVis1 = allDATAPol[s][mask,j,POL_ORDER.index('RL')]
            mask[:] = np.logical_and(mask1,allWgt[s][POL_ORDER.index('LR')][:,j]>0.0)    
            ResVis2 = allDATAPol[s][mask,j,POL_ORDER.index('LR')]
            Qplot = allQ[s][mask]/1.e6
            if len(ResVis1)>0:
              foundData = True
              CurrMax = np.max(np.abs(ResVis1))
              ss = sub.scatter(ResVis1.real,ResVis1.imag,vmin=0.,vmax=Qmax,c=Qplot,edgecolors='none',marker='.')
            if len(ResVis2)>0:
              foundData = True
              CurrMax = np.max(np.abs(ResVis2))
              ss = sub.scatter(ResVis2.real,ResVis2.imag,vmin=0.,vmax=Qmax,c=Qplot,edgecolors='none',marker='.')
            del ResVis1, ResVis2
        if foundData:
          if not plotted:
            plotted=True
            try:
              cb = pl.colorbar(ss) 
              cb.set_label(r'Baseline Length ($10^3$ km)')
            except:
              printMsg('WARNING: Problem with colorbar!',LOGNAME)
          sub.set_title('RL & LR RESIDS (Baselines to %s)'%ANTNAMES[i])
          sub.set_xlabel(r'Real $V_{res}$ (Jy)')
          sub.set_ylabel(r'Imag $V_{res}$ (Jy)')
          TotMax = np.max([MaxRes[i],CurrMax])
          if TotMax>0.0:
            sub.set_xlim((-TotMax,TotMax))
            sub.set_ylim((-TotMax,TotMax))
          sub.plot(np.array([0.,0.]),np.array([-TotMax,TotMax]),':k')
          sub.plot(np.array([-TotMax,TotMax]),np.array([0.,0.]),':k')

          pl.savefig('%s_PolSolve_Residuals_%s.png'%('.'.join(vis.split('.')[:-1]),ANTNAMES[i]))


    ifile = open('%s.spw_%s.PolSolve.CovMatrix'%(vis,spwstr.replace(',','_').replace('~','-')),'w')
    print('!  PARAMETERS:',file=ifile)



# Print final estimates of source polarization and Dterms:
    i = 0
    printMsg('\nFitting results:',LOGNAME)
    for l in range(NCalFields):
      printMsg('FIELD ID %i:\n'%l,LOGNAME)
      for si in range(NmodComp[l]):
        if FITSOU[l][si]:
          if Dofrac_pol:
            Pr = (np.sin(LM[0][i])+1.)*bound_frac_pol/2.
            if np.abs(Errors[i])>0.0:
              ErrPr = np.std((np.sin(np.random.normal(LM[0][i],np.abs(Errors[i]),1000))+1.))*bound_frac_pol/2.
            else:
              ErrPr = 0.0
            EV = (LM[0][i+1]*180./np.pi)%360.
            if EV>180.:
              EV -= 360.
            elif EV<-180.:
              EV += 360.
            ErrEV = np.abs(Errors[i+1])*180./np.pi
            Q = Pr*np.cos(EV*2.)
            U = Pr*np.sin(EV*2.)
            ErrQ = 0.0  ; ErrU = 0.0
            print(' %i -> Field %i (%s); Component %i; frac_pol: %.5e +/- %.2e .'%(i,l, SNAME[l], si,Pr,ErrPr),file=ifile)
            print(' %i -> Field %i (%s); Component %i; EVPA: %.5e +/- %.2e deg.'%(i+1,l,SNAME[l], si,EV,ErrEV),file=ifile)
            i += 2
          else:
            Q = LM[0][i]
            U = LM[0][i+1]
            ErrQ = Errors[i] ; ErrU = Errors[i+1]
            print(' %i -> Field %i (%s); Component %i; Stokes Q: %.5e +/- %.2e (frac)'%(i,l, SNAME[l], si,Q,ErrQ),file=ifile)
            print(' %i -> Field %i (%s); Component %i; Stokes U: %.5e +/- %.2e (frac)'%(i+1,l,SNAME[l], si,U,ErrU),file=ifile)
            i += 2
      # Unbiased estimates and errors (quick&dirty MonteCarlo):
            Nmc = 1000
            Qmc = np.random.normal(Q,ErrQ,Nmc) ; Umc = np.random.normal(U,ErrU,Nmc)
            Pmc = np.sqrt(Qmc*Qmc + Umc*Umc) ; EVmc = np.arctan2(Umc,Qmc)*90./np.pi

            for mi in range(Nmc):
              dAng = EVmc[mi] - EVmc[0]
              if dAng > 180.:
                EVmc[mi] -= 180.
              elif dAng < -180.:
                EVmc[mi] += 180.
          
            Pr = np.average(Pmc)
            ErrPr = np.std(Pmc)

            EV = np.average(EVmc)
            ErrEV = np.std(EVmc)

          if DoFaraday:
            RM = LM[0][i] ; RMErr = Errors[i]
            spix = 2.*LM[0][i+1] ; spixErr = 2.*Errors[i+1]
            print(' %i -> Field %i (%s); Component %i; RM: %.5e +/- %.2e rad/m^2.'%(i,l, SNAME[l], si,RM,RMErr),file=ifile)
            print(' %i -> Field %i (%s); Component %i; Pol. Spix: %.5e +/- %.2e .'%(i+1,l,SNAME[l], si,spix,spixErr),file=ifile)
            i += 2


        else:
          Pr = cfrac_pol[l][si]
          EV = cEVPA[l][si]
          Q = Pr*np.cos(EV*np.pi/90.)
          U = Pr*np.sin(EV*np.pi/90.)
          ErrPr = 0.0 ; ErrEV = 0.0 ; ErrQ = 0.0  ; ErrU = 0.0
        printMsg('Source id %i (%s), Component #%i: frac_pol = % 6.4f +/- %.3f ; EVPA = % 6.2f +/- %.2f deg. (Q/I = %.3e +/- %.2e ; U/I = %.3e +/- %.2e )'%(calField[l], SNAME[l], si,Pr,ErrPr,EV,ErrEV,Q,ErrQ,U,ErrU),LOGNAME)



    printMsg('\n Dterms (Right):',LOGNAME)
    for si in range(nant):
      if DRSolve[si]:
        Re = LM[0][i] ; Im = LM[0][i+1]
        ErrRe = Errors[i]; ErrIm = Errors[i+1]
        DRa[si,0] = Re + 1.j*Im
        ErrR[si,0] = np.sqrt(ErrRe*ErrIm)
        print(' %i -> Antenna %i (%s); Dterm R (real): %.5e +/- %.2e'%(i,si,anam[si],Re,ErrRe),file=ifile)
        print(' %i -> Antenna %i (%s); Dterm R (imag): %.5e +/- %.2e'%(i+1,si,anam[si],Im,ErrIm),file=ifile)
        if nterms==1 or not FitPerIF:
          printMsg('Antenna #%i (%s):  Real = % .2e +/- %.1e; Imag = % .2e +/- %.1e | Amp = %.4f | Phase: %.2f deg.'%(si,anam[si], Re,ErrRe,Im,ErrIm, np.sqrt(Re*Re+Im*Im),np.arctan2(Im,Re)*180./np.pi),LOGNAME)

        if nterms>1:
          for ii in range(1,nterms):
            Re = LM[0][i+2+(ii-1)*2] ; Im = LM[0][i+3+(ii-1)*2]
            ErrRe = Errors[i+2+(ii-1)*2]; ErrIm = Errors[i+3+(ii-1)*2]
            DRa[si,ii] = Re + 1.j*Im
            print(' %i -> Antenna %i; Dterm R Nu-power %i (real): %.5e +/- %.2e'%(i+2+(ii-1)*2,si,ii,Re,ErrRe),file=ifile)
            print(' %i -> Antenna %i; Dterm R Nu-power %i (imag): %.5e +/- %.2e'%(i+3+(ii-1)*2,si,ii,Im,ErrIm),file=ifile)
            ErrR[si,ii] = np.sqrt(ErrRe*ErrIm)
          if FitPerIF:   
            Re = np.median(DRa[si,:].real) #,weights=1./ErrR[si,:]**2.); 
            Im = np.median(DRa[si,:].imag) #,weights=1./ErrR[si,:]**2.); ErrRe = np.average(ErrR[si,:])
            printMsg('Antenna #%i (%s):  Real = % .2e +/- %.1e; Imag = % .2e +/- %.1e | Amp = %.4f | Phase: %.2f deg.'%(si,anam[si], Re,ErrRe,Im,ErrRe, np.sqrt(Re*Re+Im*Im),np.arctan2(Im,Re)*180./np.pi),LOGNAME)
   
        i += 2*nterms

      else:
        Re = DRa[si,0].real ; Im = DRa[si,0].imag
        ErrRe = 0.0 ; ErrIm = 0.0
        ErrR[si,:] = 0.0
        printMsg('Antenna #%i (%s):  Real = % .2e +/- %.1e; Imag = % .2e +/- %.1e | Amp = %.4f | Phase: %.2f deg.'%(si,anam[si], Re,ErrRe,Im,ErrIm, np.sqrt(Re*Re+Im*Im),np.arctan2(Im,Re)*180./np.pi),LOGNAME)


    printMsg('\n Dterms (Left):',LOGNAME)
    for si in range(nant):
      if isLinear[si]:
         print('Antenna %i (%s); Dterm L = Dterm R'%(si,anam[si]),file=ifile)
         DLa[si,:] = DRa[si,:]

      else:

          if DLSolve[si]:
            Re = LM[0][i] ; Im = LM[0][i+1]
            ErrRe = Errors[i]; ErrIm = Errors[i+1]
            DLa[si,0] = Re + 1.j*Im
            ErrL[si,0] = np.sqrt(ErrRe*ErrIm)
            print(' %i -> Antenna %i (%s); Dterm L (real): %.5e +/- %.2e'%(i,si,anam[si],Re,ErrRe),file=ifile)
            print(' %i -> Antenna %i (%s); Dterm L (imag): %.5e +/- %.2e'%(i+1,si,anam[si],Im,ErrIm),file=ifile)
            if nterms==1 or not FitPerIF:
              printMsg('Antenna #%i (%s):  Real = % .2e +/- %.1e; Imag = % .2e +/- %.1e | Amp = %.4f | Phase: %.2f deg.'%(si,anam[si], Re,ErrRe,Im,ErrIm, np.sqrt(Re*Re+Im*Im),np.arctan2(Im,Re)*180./np.pi),LOGNAME)
            if nterms>1:
              for ii in range(1,nterms):
                Re = LM[0][i+2+(ii-1)*2] ; Im = LM[0][i+3+(ii-1)*2]
                ErrRe = Errors[i+2+(ii-1)*2]; ErrIm = Errors[i+3+(ii-1)*2]
                DLa[si,ii] = Re + 1.j*Im
                print(' %i -> Antenna %i; Dterm L Nu-power %i (real): %.5e +/- %.2e'%(i+2+(ii-1)*2,si,ii,Re,ErrRe),file=ifile)
                print(' %i -> Antenna %i; Dterm L Nu-power %i (imag): %.5e +/- %.2e'%(i+3+(ii-1)*2,si,ii,Im,ErrIm),file=ifile)
                ErrL[si,ii] = np.sqrt(ErrRe*ErrIm)
              if FitPerIF:   
                Re = np.median(DLa[si,:].real) #,weights=1./ErrL[si,:]**2.); 
                Im = np.median(DLa[si,:].imag) #,weights=1./ErrL[si,:]**2.); ErrRe = np.average(ErrR[si,:])
                printMsg('Antenna #%i (%s):  Real = % .2e +/- %.1e; Imag = % .2e +/- %.1e | Amp = %.4f | Phase: %.2f deg.'%(si,anam[si], Re,ErrRe,Im,ErrRe, np.sqrt(Re*Re+Im*Im),np.arctan2(Im,Re)*180./np.pi),LOGNAME)
         
            i += 2*nterms

          else:
            Re = DLa[si,0].real ; Im = DLa[si,0].imag
            ErrRe = 0.0 ; ErrIm = 0.0
            ErrL[si,:] = 0.0
            printMsg('Antenna #%i (%s):  Real = % .2e +/- %.1e; Imag = % .2e +/- %.1e | Amp = %.4f | Phase: %.2f deg.'%(si,anam[si], Re,ErrRe,Im,ErrIm, np.sqrt(Re*Re+Im*Im),np.arctan2(Im,Re)*180./np.pi),LOGNAME)


# Write CovMatrix:
    print('!  POST-FIT COVARIANCE MATRIX:',file=ifile)

    fmt = '%.2e '*Npar
    for i in range(Npar):
      print(fmt%tuple(LM[1][i,:]),file=ifile)

    ifile.close()
  
  

# Write calibration table:
    form = '%.1f  % i  %i  %i  % i  %.1f  % i  % i  % .8f  % .8f  % .8f  % .8f  % .8f  % .8f  %i  %i  %.3f  %.3f\n'
    DtName = '%s.spw_%s.Dterms'%(vis,spwstr.replace(',','_').replace('~','-'))
    if os.path.exists(DtName):
      os.system('rm -rf %s'%(DtName))
    ascf = open('%s.Dterms.ascii'%vis,'w')



    ascf.write('TIME FIELD_ID %s ANTENNA1 ANTENNA2 INTERVAL SCAN_NUMBER OBSERVATION_ID CPARAM PARAMERR FLAG SNR\n'%SPIDNAME)
    ascf.write('D I I I I D I I X2,1 R2,1 B2,1 R2,1\n')


    for j in range(nant):
  #  for spi in spwl:
      if FitPerIF:
        for spi in spwl:
          ascf.write(form%(0,-1,spi,j,-1,0,-1,-1, DRa[j,spi].real, DRa[j,spi].imag, DLa[j,spi].real, DLa[j,spi].imag,ErrR[j,spi], ErrL[j,spi],0 ,0, 100., 100.))
      else:
        ascf.write(form%(0,-1,int(spi),j,-1,0,-1,-1, DRa[j,0].real, DRa[j,0].imag, DLa[j,0].real, DLa[j,0].imag,ErrR[j,0], ErrL[j,0],0 ,0, 100., 100.))


    ascf.close()
    tb.fromascii(tablename=DtName,asciifile='%s.Dterms.ascii'%vis,sep=' ')
    tb.close()
    tb.open(DtName,nomodify=False)
    tb.putinfo({'readme': '', 'subType': 'D Jones', 'type': 'Calibration'})
    tb.close()


    if not FitPerIF:
      for i in range(1,nterms):

        DtName = '%s.spw_%s_p%i.Dterms'%(vis,spwstr.replace(',','_').replace('~','-'),i)
        if os.path.exists(DtName):
          os.system('rm -rf %s'%(DtName))
        ascf = open('%s.Dterms.ascii'%vis,'w')



        ascf.write('TIME FIELD_ID %s ANTENNA1 ANTENNA2 INTERVAL SCAN_NUMBER OBSERVATION_ID CPARAM PARAMERR FLAG SNR\n'%SPIDNAME)
        ascf.write('D I I I I D I I X2,1 R2,1 B2,1 R2,1\n')


        for j in range(nant):
     # for spi in spwl:
          ascf.write(form%(0,-1,int(spi),j,-1,0,-1,-1, DRa[j,i].real, DRa[j,i].imag, DLa[j,i].real, DLa[j,i].imag,ErrR[j,0], ErrL[j,0],0 ,0, 100., 100.))



        ascf.close()
        tb.fromascii(tablename=DtName,asciifile='%s.Dterms.ascii'%vis,sep=' ')
        tb.close()
        tb.open(DtName,nomodify=False)
        tb.putinfo({'readme': '', 'subType': 'D Jones', 'type': 'Calibration'})
        tb.close()


### TODO: getChi2!!!




#  li = len(allDATAPol)
#  for it in range(li):
#    del allDATAPol[li-it-1]
#  del allDATAPol

 # li = len(allWgt)
 # for it in range(li):
 #   for j in range(4):
 #     del allWgt[li-it-1][3-j]
 #   del allWgt[li-it-1]
 # del allWgt

 # li = len(WgtCorr)
 # for it in range(li):
 #   del WgtCorr[li-it-1]
 # del WgtCorr

#  li = len(allEMA)
#  for it in range(li):
#    del allEMA[li-it-1]
#    del allEPA[li-it-1]
#    del allAN1[li-it-1]
#    del allAN2[li-it-1]
#    del allSPW[li-it-1]
#    del allCOMPS[li-it-1]
#  del allEMA,allEPA,allAN1,allAN2,allSPW,allCOMPS

    for i in range(len(allDATAPol)-1,-1,-1):
      del allDATAPol[i], allEMA[i], allEPA[i],allAN1[i],allAN2[i],allSPW[i]
      for j in range(3,-1,-1):
        del allWgt[i][j]

      if NCalFields>0:
        del allCOMPS[i]
      else:
        allMODELPol[i]



    gc.collect()














  targets = str(target_field)
  if len(targets)>0:
    selFields,selFieldNames = getFields(vis,targets)
  #  if not parang_corrected:
  #    printError('IN THE CURRENT VERSION, DTERMS ARE ONLY APPLIED IF parang_corrected is True!')
  else:
    selFields = []
    selfFieldNames = []
 

  if len(selFields)==0:
    return
   # raw_input('HOLD')

  tb.open(os.path.join(vis,'SPECTRAL_WINDOW'))
  Nchan = float(tb.getcol('NUM_CHAN')[0])
  tb.close()

  metaDATA = {}

  tb.open(os.path.join(vis,'FIELD'))
  scoord = tb.getcol('PHASE_DIR')
  RA = scoord[0,0,:]
  Dec = scoord[1,0,:]
  tb.close()

  tb.open(os.path.join(vis,'ANTENNA'))
  apos = tb.getcol('POSITION')
  metaDATA['apos'] = apos
  tb.close()


  DRaNu = {} ; DLaNu = {}
  for spi,spw in enumerate(spwl):
    DRaNu[int(spw)] = np.zeros((int(nant),int(Nchan)),dtype=np.complex128)
    DLaNu[int(spw)] = np.zeros((int(nant),int(Nchan)),dtype=np.complex128)

    if FitPerIF:
      DRaNu[spw][:] = -DRa[:,spi][:,np.newaxis]
      DLaNu[spw][:] = -DLa[:,spi][:,np.newaxis]
    else:
      DRaNu[spw] -= DRa[:,0][:,np.newaxis]
      DLaNu[spw] -= DLa[:,0][:,np.newaxis]
      for i in range(1,nterms):
        DRaNu[spw] -= DRa[:,i][:,np.newaxis]*FreqPow[i-1][:,spi][np.newaxis,:]
        DLaNu[spw] -= DLa[:,i][:,np.newaxis]*FreqPow[i-1][:,spi][np.newaxis,:]
  


#  toplot = np.array([DRaNu[i][0,0] for i in range(len(spwl))])
#  pl.plot(-toplot.real, 'ob')
#  pl.plot(-toplot.imag, 'or')
#  toplot = np.array([DLaNu[i][0,0] for i in range(len(spwl))])
#  pl.plot(-toplot.real, 'sb')
#  pl.plot(-toplot.imag, 'sr')

#  pl.show() ; raw_input('HOLD')


  SFac = {}
  for sp in spwl:
    SFac[sp] = 1./(1.-DRaNu[sp]*DLaNu[sp])  

  tb.open(vis,nomodify=False)

  SPIDNAME = 'SPECTRAL_WINDOW_ID'
  if SPIDNAME not in list(tb.colnames()):
    SPIDNAME = 'DATA_DESC_ID'


  CalDATA = {}

  for inputs in ['DATA','ANTENNA1','ANTENNA2','TIME','FIELD_ID']:
    CalDATA[inputs.lower()] = tb.getcol(inputs)

  CalDATA2 = tb.getcol('CORRECTED_DATA')

  for spi,spw in enumerate(spwl):
   selSPW = tb.getcol(SPIDNAME)==int(spw) 

   for target in selFields:
    printMsg('\nApplying calibration to field id %i (spw %i)'%(target,spw),LOGNAME)  
    selCol = selSPW*(CalDATA['field_id']==target)

    DATA = {'data':np.copy(CalDATA['data'][:,:,selCol]), 'antenna1':np.copy(CalDATA['antenna1'][selCol]), 
            'antenna2':np.copy(CalDATA['antenna2'][selCol]), 'time':np.copy(CalDATA['time'][selCol])}

    metaDATA['RA'] = RA[target]
    metaDATA['Dec'] = Dec[target]
  
    PAs = getParangle(metaDATA,DATA, mounts, FeedAngles)  

# Sum and difference of PANGS (in complex form):
    EPA = np.exp(1.j*(PAs[:,0]+PAs[:,1])) ; EMA = np.exp(1.j*(PAs[:,0]-PAs[:,1]))

# Get data back into antenna frame:
    if parang_corrected:
      CalDATA2[polprods.index('RR'),:,selCol] = CalDATA['data'][polprods.index('RR'),:,selCol]*EMA[:,np.newaxis]
      CalDATA2[polprods.index('RL'),:,selCol] = CalDATA['data'][polprods.index('RL'),:,selCol]*EPA[:,np.newaxis]
      CalDATA2[polprods.index('LR'),:,selCol] = CalDATA['data'][polprods.index('LR'),:,selCol]/EPA[:,np.newaxis]
      CalDATA2[polprods.index('LL'),:,selCol] = CalDATA['data'][polprods.index('LL'),:,selCol]/EMA[:,np.newaxis]

# Apply Dterms:
    SDt = SFac[spw][DATA['antenna1'],:]*np.conjugate(SFac[spw][DATA['antenna2'],:])
    BKP_RR = np.copy(CalDATA2[polprods.index('RR'),:,selCol],order='C')*SDt 
    BKP_RL = np.copy(CalDATA2[polprods.index('RL'),:,selCol],order='C')*SDt 
    BKP_LR = np.copy(CalDATA2[polprods.index('LR'),:,selCol],order='C')*SDt 
    BKP_LL = np.copy(CalDATA2[polprods.index('LL'),:,selCol],order='C')*SDt 

    CalDATA2[polprods.index('RR'),:,selCol] = BKP_RR + np.conjugate(DRaNu[spw][DATA['antenna2'],:])*BKP_RL + DRaNu[spw][DATA['antenna1'],:]*BKP_LR + DRaNu[spw][DATA['antenna1'],:]*np.conjugate(DRaNu[spw][DATA['antenna2'],:])*BKP_LL
    CalDATA2[polprods.index('RL'),:,selCol] = BKP_RL + DRaNu[spw][DATA['antenna1'],:]*np.conjugate(DLaNu[spw][DATA['antenna2'],:])*BKP_LR + DRaNu[spw][DATA['antenna1'],:]*BKP_LL + np.conjugate(DLaNu[spw][DATA['antenna2'],:])*BKP_RR
    CalDATA2[polprods.index('LR'),:,selCol] = BKP_LR + DLaNu[spw][DATA['antenna1'],:]*np.conjugate(DRaNu[spw][DATA['antenna2'],:])*BKP_RL + DLaNu[spw][DATA['antenna1'],:]*BKP_RR + np.conjugate(DRaNu[spw][DATA['antenna2'],:])*BKP_LL
    CalDATA2[polprods.index('LL'),:,selCol] = BKP_LL + np.conjugate(DLaNu[spw][DATA['antenna2'],:])*BKP_LR + DLaNu[spw][DATA['antenna1'],:]*BKP_RL + DLaNu[spw][DATA['antenna1'],:]*np.conjugate(DLaNu[spw][DATA['antenna2'],:])*BKP_RR

# Put data into sky frame:
    CalDATA2[polprods.index('RR'),:,selCol] /= EMA[:,np.newaxis]
    CalDATA2[polprods.index('RL'),:,selCol] /= EPA[:,np.newaxis]
    CalDATA2[polprods.index('LR'),:,selCol] *= EPA[:,np.newaxis]
    CalDATA2[polprods.index('LL'),:,selCol] *= EMA[:,np.newaxis]

    allKeys = [hdp for hdp in DATA.keys()]
    for inputs in allKeys: #DATA.keys():
        del DATA[inputs]
    del DATA
  #  del DATA['data'], DATA['antenna1'], DATA['antenna2'], DATA['time']

# Save data:
  tb.putcol('CORRECTED_DATA',CalDATA2)
  tb.unlock()
  tb.close()
   

# Release memory:
  if len(selFields)>0:
    del BKP_RR, BKP_LL, BKP_RL, BKP_LR, CalDATA2
    allKeys = [hdp for hdp in CalDATA.keys()]
    for inputs in allKeys: # CalDATA.keys():
      del CalDATA[inputs.lower()]
    del CalDATA

  gc.collect()
    

## END!!

  print('DONE!')


if __name__=='__main__':

  polsolve(vis, spw, field, mounts, DR, DL, DRSolve, DLSolve, 
           CLEAN_models, frac_pol, EVPA, PolSolve, parang_corrected,target_field)


