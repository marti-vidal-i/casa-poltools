# (c) I. Marti-Vidal (Univ. Valencia) 2020.

# Script to simulate full-polarization EHT observations of a 
# resolved calibrator.
# This script complements the contents of the paper:

# I. Marti-Vidal et al. 2021, A&A, 646, 52




from task_polsimulate import polsimulate

import numpy as np
import os
import pickle as pk

# Set random seed (to get the same values as in the paper):
np.random.seed(42)


# Get random Dterms from a Gaussian distribution:
DRs = (np.random.normal(0.,0.20,8)+np.random.normal(0.,0.20,8)*1.j)
DLs = (np.random.normal(0.,0.20,8)+np.random.normal(0.,0.20,8)*1.j)


# True means to re-generate the synthetic data:
DOSIM = False

# True means to re-CLEAN (manually) all four Stokes parameters:
# The mask made from this step will be used later in the 
# "polarization self-calibration" steps:
DOCLEAN = True




if DOSIM:
 os.system('rm -rf SgrA_polsimulate*')
 polsimulate(vis='SgrA_polsimulate.ms',
  reuse=False, # A new Measurement Set will be made from scratch.
  model_Dt_0 = DRs, # Dterms in first polarizer (i.e., R)
  model_Dt_1 = DLs, # Dterms in second polarizer (i.e., L)
  feed = 'circular', # The polarizers are actually R and L.
  I = [1.0,0.8,0.6,0.4,0.4], # Stokes I of the 5 source components.
  Q_frac = [0.,0.,0.,0.8,0.0], # Fractional Stokes Q.
  U_frac = [0.,0.,0.,0.5,-0.9], # Fractional Stokes U.
  V_frac = [0.,0.,0.,0.,0.], # Fractional V
  RM = [0.,0.,0.,0.,0.],  # Rotation measures.
  spec_index = [0.,0.,0.,0.,0.], # Spectral indices.
## RA offsets of each of the five source components (given in degrees):
  RAoffset = [0.,-40.e-6/3600.,-80.e-6/3600.,-10.e-6/3600.,-30.e-6/3600.],
## Samce for the Dec offsets (all are zero, since the jet is in E-W direction:
  Decoffset = [0.,0.,0.,0.,0.],
## Phase center of the source core (that of SgrA*)
  phase_center = "J2000 17h45m40.4230 -29d00m28.0400",
  array_configuration    =  "EHT.cfg", # ascii file with the antenna info.
## Antenna mounts (the same order as the antenna config. file is assumed):
  mounts  =  ['AZ', 'NR', 'NR', 'AZ', 'NL', 'NL', 'AZ', 'NL'],
  LO = 2.3e+11, BBs = [0.0], spw_width = 2.e9, nchan = 1, # Freq. config.
  nscan = "TRACK_C.listobs", # Take observing times from this listobs file.
  visib_time = '6s', # VLBI integration time.
  apply_parang=True, # Apply parallactic angle to the data.
  corrupt=True) # Add noise using the "sm" tool algorithm.



# Just CLEAN with the "tclean" task. Don't forget to create
# a manual mask **AND APPLY IT TO ALL POLARIZATION CHANNELS**:
if DOCLEAN:
 os.system('rm -rf SgrA_clean*')
 clearcal(vis='SgrA_polsimulate.ms',addmodel=True)
 tclean(vis='SgrA_polsimulate.ms',
    imagename='SgrA_clean',
    specmode = 'mfs',
    niter = 2000,
    interactive=True,
    imsize=1024,
    cell = '0.000001arcsec',
    stokes = 'IQUV',
    weighting = 'uniform',
    deconvolver='hogbom',
    gain=0.1,
    restart=False,
    savemodel = 'modelcolumn')

# Change the name of the mask (to keep it!):
 os.system('rm -rf TCLEAN.mask')
 os.system('cp -r SgrA_clean.mask TCLEAN.mask')



# Write the simulated Dterms values into an external file:
outp = open('EHT_Dterms.dat','wb')
pk.dump([DRs,DLs], outp)
outp.close()

 
