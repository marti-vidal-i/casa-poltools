# (c) I. Marti-Vidal (Univ. Valencia) 2020.

# Script to solve for the instrumental polarization of (synthetic)
# EHT observations of a resolved calibrator, using the "polarization
# self-calibration" strategy (described in Cotton 1993).
# This script complements the contents of the paper:

# I. Marti-Vidal et al. 2021, A&A, 646, 52


from task_polsolve import polsolve

import numpy as np
import pylab as pl
import os,sys


# Number of self-cal iterations:
NITER = 10

# First, we re-fill the "corrected" column of the measurement set 
# with the original data (to undo any previous calibration):
clearcal(vis='SgrA_polsimulate.ms',addmodel=True)



# ITERATE!!
for i in range(NITER):

  print('\n\n ITERATION %i\n\n'%i)

## We CLEAN the image, using the visibilities stored in 
# the "corrected" column of the measurement set:

  os.system('rm -rf SgrA_SC_clean*')
  os.system('rm -rf TEST.ms')
  split(vis='SgrA_polsimulate.ms',outputvis='TEST.ms',datacolumn='corrected')
#  clearcal(vis='TEST.ms',addmodel=True)

  tclean(vis='TEST.ms',
    imagename='SgrA_SC_clean',
    specmode = 'mfs',
    niter = 200,
    interactive=False,
    imsize=1024,
    cell = '0.000001arcsec',
    stokes = 'IQUV',
    mask = 'TCLEAN.mask', ## We use the mask that we created manually.
    weighting = 'uniform',
    deconvolver='hogbom',
    gain=0.1,
    restart=False)

# We remove the previous Dterm table:
  os.system('rm -rf SgrA_polsimulate.ms.spw_0.Dterms')

# Solve for the Dterms:
  polsolve(vis='SgrA_polsimulate.ms',
## If we set "target_field", the Dterms will not only be 
# estimated, but also APPLIED to the "corrected" column.
# This is ESSENTIAL for the self-cal approach to work: 
         target_field='POLSIM', # Source name
## Antenna mounts (the ordering should be that of the ANTENNA table):
         mounts=['AZ', 'NR', 'NR', 'AZ', 'NL', 'NL', 'AZ', 'NL'],
## If "CLEAN_models" is the path to a (full-polarization) CASA image,
## the self-calibration approach is activated:
         CLEAN_models='SgrA_SC_clean.model',
# These values are not really used:
         frac_pol=[0.], EVPA=[0.], PolSolve=[False],
## Set this to True, if you want to test the linear Dterm model:
         linear_approx=False)

## Save the Dterm table with a name that contains the iteration number:
  os.system('rm -rf ITER_%i.Dterms'%i)
  os.system('cp -r SgrA_polsimulate.ms.spw_0.Dterms ITER_%i.Dterms'%i)
  tb.open('SgrA_polsimulate.ms')
  tb.unlock()
  tb.close()


## THAT'S IT! After the last iteration, the final data should se stored 
# in the "corrected" column of the measurement set.



## READ FITTED DTERMS:
tb.open('ITER_%i.Dterms'%i)
DTS = tb.getcol('CPARAM')
tb.close()



## RECOVER SIMULATED DTERMS:

# Set random seed (to get the same values as in the paper):
np.random.seed(42)


# Get random Dterms from a Gaussian distribution:
DRs = (np.random.normal(0.,0.20,8)+np.random.normal(0.,0.20,8)*1.j)
DLs = (np.random.normal(0.,0.20,8)+np.random.normal(0.,0.20,8)*1.j)



#### PLOT CORRELATION OF DTERMS:
fig = pl.figure()
sub = fig.add_subplot(111)
for ant in range(8):
  sub.plot(DRs.real,DTS[0,0,:].real,'or')
  sub.plot(DRs.imag,DTS[0,0,:].imag,'sr')
  sub.plot(DLs.real,DTS[1,0,:].real,'ob')
  sub.plot(DLs.imag,DTS[1,0,:].imag,'sb')

sub.plot([],[],'or',label='DR Real')
sub.plot([],[],'sr',label='DR Imag')
sub.plot([],[],'ob',label='DL Real')
sub.plot([],[],'sb',label='DL Imag')
sub.plot(np.array([-0.5,0.5]),np.array([-0.5,0.5]),':k')
pl.legend(numpoints=1,loc=4)
pl.xlabel('True Dterms')
pl.ylabel('PolSolve Dterms')
pl.suptitle('Pol. SelfCal')
pl.savefig('DTERM_COMPARISON_SELF-CAL.png')
pl.show()










