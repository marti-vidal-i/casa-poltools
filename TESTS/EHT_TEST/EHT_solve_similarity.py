# (c) I. Marti-Vidal (Univ. Valencia) 2020.

# Script to solve for the instrumental polarization of (synthetic)
# EHT observations of a resolved calibrator, using the "source slicing".
# strategy, as originally implemented in LPCAL (Leppanen 1995).
# This script complements the contents of the paper:

# I. Marti-Vidal, A. Mus, P. de Vicente & J. Gonzalez (2020), ... TBC





import numpy as np
import pylab as pl
import os, sys
import glob

# The user should use the task "imview" to get the pixel coordinates of the
# BLC and TRC of the boxes that will define the polarization sub-components.
# In this case, we are using several sub-components, stored in the "MyRegions.rg" 
# file. The helper task called "CCextract" takes all the CASA (or DS9) regions 
# defined there (you can use  boxes, circles, ellipses, etc.), extracts the 
# CC components of each region and saves them in ascii format, to be read by 
# PolSolve:
CCextract(model_image='SgrA_clean.model',
          regions='MyRegions_3cmp.rg',make_plot=True)

## Get all the subcomponent files:
ALLCCs = glob.glob('SgrA_clean.model.CC??')

# NOTE: CCextract has an option to make a plot of the CLEAN deltas in each
# sub-component (using different colors). In addition, you should pay attention
# to the warnings regarding CLEAN components appearing in more than one region.
# If this happens, the CLEAN component is only added to the region first 
# appearing in the list. This behaviour can indeed be used to make fancy 
# boolean region conditions!


# Solve for Dterms and source polarization:
polsolve(vis='SgrA_polsimulate.ms',
## The mounts should be given following the order of the ANTENNA table:
         mounts=['AZ', 'NR', 'NR', 'AZ', 'NL', 'NL', 'AZ', 'NL'],
## This is the list of ascii files created by "CCextract":
         CLEAN_models=ALLCCs,
## Initial guesses of the polarization state of each sub-component:
         frac_pol=[0. for pi in ALLCCs], EVPA=[0. for pi in ALLCCs], 
## Solve for the polarization of all three sub-components:
         PolSolve=[True for pi in ALLCCs],
## Apply the full non-linear Dterm model:
         linear_approx=False)

## Change the name of the Dterm CASA table, to keep it:
os.system('rm -rf LPCAL.Dterms')
os.system('cp -r SgrA_polsimulate.ms.spw_0.Dterms LPCAL.Dterms')

## READ FITTED DTERMS:
tb.open('SgrA_polsimulate.ms.spw_0.Dterms')
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
pl.suptitle('SubComponent Fitting')
pl.savefig('DTERM_COMPARISON_SELF-SIMILARITY.png')
pl.show()










