################################################################
# Helper functions for polarization calibration & plotting
# Version "May 4, 2022 v1" - I. Marti-Vidal
# If you use any of these functions, please refer to:
#
#         Marti-Vidal et al. A&A, 646, A52
#
################################################################

import matplotlib.pyplot as plt
import os
import pylab as pl
import numpy as np
import pyfits as pf
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
from itertools import combinations

from casatools import image #as ia
from casatools import table
from casatasks import imhead
from casatools import ms as mset

ia = image()
tb = table()
ms = mset()

#from taskinit import gentools
#ia = gentools(['ia'])[0]




##########################
### HELPER FUNCTIONS INCLUDED SO FAR:
#
#     plotPolImage
#     plotDterms
#     plotPolCovMatrix
#     plotPolTraces
##########################




# Load LaTeX fonts:
if True:
  plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
  params = {'text.usetex' : True,
          'font.size' : 18,
          'font.family' : 'lmodern',
      #    'text.latex.unicode': True,
          }
  plt.rcParams.update(params) 
  





# Plot a full-stokes image as mpol+EVPA image.
def plotPolImage(img, mSigmaCut = 5., SigmaCut = 3., dPol = 20, lPol = 0.05, 
                figName='polImg.png', zoom = 1.0, doEVPA = True, zoomCenter = [0,0], 
                vmax=0.0, figUnit='mas', polUnit = 'mJy/beam', levs=10, title=''):
  """ 
  Generates a publication-ready full-polarization image.
  
  PARAMTERS:  
      ``img'' :: name of the (t)CLEAN CASA image ("IQUV").
      ``title'' :: Title of the figure (default is source name with info).
      ``mSigmaCut'' :: noise cutoff for the linear-polarization image (in units of the rms).
      ``SigmaCut'' :: noise cutoff for Stokes I (in units of rms).
      ``levs'' :: Contour levels for Stokes I. If integer, number of contours in logspace (starting a SigmaCut); if list of numbers, contour levels in units of image peak.
      ``dPol'' :: distance between EVPA vectors, in pixel units.
      ``lPol'' :: maximum length of the EVPA vectors (in image-size units).
      ``figName'' :: name of final image (with desired extension)
      ``vmax'' :: Peak fractional polarization (if 0, will take it from the data).
      ``zoom'' :: Zooming factor (toward the zoomCenter pixel).
      ``zoomCenter'' :: zoom shift with respect to image center (in pixel units). Default is the center of the image.
      ``figUnit'' :: axes unit (can be "muas", "mas", "arcsec", "arcmin", "deg" or "rad").

  RETURNS:
      The matplotlib figure instance (so that the user can fine-tune it further).

  """



  print('This function is part of the CASA-poltools.\n    Please refer to:   Marti-Vidal et al. A&A, 646, A52')

# Parameters for image plot(s):
  beamcolor = 'b'

  Kfac = {'muas':1.e-6,'as':1.,'mas':1.e-3,'arcmin':60.,'arcsec':1.,'deg':3600., 'rad':180./np.pi*3600.}

  PixPerVec = dPol
  SNR_CUT = mSigmaCut
  SNR_CUT_I = SigmaCut


  ia.close()

  imdata = {}

  if os.path.exists(img):
    try: 
      ia.open(img)
    except:
      raise Exception('PlotImage ERROR: Bad CASA image \'%s\'!'%img)
  else:
    raise Exception('PlotImage ERROR: Unexistent CASA image \'%s\'!'%img)


  TEMPDATA = ia.getchunk()

  for si,STK in enumerate(['I','Q','U','V']):
    imdata[STK]= np.copy(TEMPDATA[:,:,si,0])
    imdata[STK][np.isnan(imdata[STK])] = 0.0

  summ = ia.summary()
  peak = np.max(imdata['I'])
  Vpeak = np.max(np.abs(imdata['V']))  

  ia.close()

  SNAME = imhead(img, mode='get',hdkey='object')

  resName = '.'.join(img.split('.')[:-1]) + '.residual'
  if os.path.exists(resName):
    ia.open(resName)
    RMS = np.std(ia.getchunk()[:,:,0,0])
    ia.close()
    print('Estimated noise (from I residuals): %.3e mJy/beam'%(1000.*RMS))
  else:  # We take it from V Stokes:
    RMS = np.std(imdata['V'])
    print('Estimated noise (from V image): %.3e mJy/beam'%(1000.*RMS))

  print('Will apply a polarization noise cutoff at %.3e mJy/beam'%(1000.*RMS*SNR_CUT))
  print('Will apply a Stokes I noise cutoff at %.3e mJy/beam'%(1000.*RMS*SNR_CUT_I))


  NPIX = list(summ['shape'])[0]
  refval = list(summ['refval'])
  refpix = list(summ['refpix'])
  incr = list(summ['incr'])
  axunit = summ['axisunits'][0]
#  IMSIZE = np.abs(incr[0])*NPIX*Kfac[axunit]/Kfac[figUnit]/2.
  pixSize = Kfac[axunit]/Kfac[figUnit]*np.abs(incr[0])
  IMSIZE = pixSize*NPIX/2.

  ArrowLen = lPol  #*NPIX


  SelPix = list(range(0,NPIX,dPol))
  SelPixOffset = [[si+j for si in SelPix] for j in range(dPol)] 
  dRA = zoomCenter[0]*pixSize ; dDec = zoomCenter[1]*pixSize

  Extent = [IMSIZE, -IMSIZE, -IMSIZE, IMSIZE]

  xc = np.linspace(IMSIZE, -IMSIZE, NPIX)
  yc = np.linspace(-IMSIZE, IMSIZE, NPIX)
  xyc = np.meshgrid(yc,xc)

  try:
    if 'restoringbeam' in summ.keys():
      M = summ['restoringbeam']['major']['value']
      m = summ['restoringbeam']['minor']['value']
      PA = summ['restoringbeam']['positionangle']['value']
    else:
      BB = summ['perplanebeams']['beams']['*0']['*0']
      M = BB['major']['value']
      m = BB['minor']['value']
      PA = BB['positionangle']['value']
    Rbeam = [M,m,PA]
  except:
    print("No beam information available!")
    Rbeam = [0.0, 0.0, 0.0] 


  m = np.sqrt(imdata['Q']**2. + imdata['U']**2.)

  m[np.isnan(m)] = 0.0

  mask = np.zeros(np.shape(m),dtype=bool)
  mask[:] = (m > RMS*SNR_CUT)*(imdata['I'] > RMS*SNR_CUT)


  XLoc, YLoc = np.where(mask)

  EVPA = np.arctan2(imdata['U'],imdata['Q'])/2.0

  polpeak = np.unravel_index(np.argmax(m), np.shape(m))



  MAXPA = EVPA[polpeak]

  mPeak = m[polpeak]
  Pfrac = mPeak/imdata['I'][polpeak]

  m[np.logical_not(mask)] = np.nan

  fig = pl.figure(figsize=(6,6))
  sub = fig.add_subplot(111)
  fig.subplots_adjust(right=0.97, top=0.93,left=0.17)

  Larray = np.linspace(RMS*SNR_CUT_I,np.max(imdata['I']),10)

  if type(levs) is int:
    if RMS>0.0:
      LEVS = list(np.power(RMS*SNR_CUT_I,np.linspace(1.,np.log(np.max(imdata['I']))/np.log(RMS*SNR_CUT),levs)))
    else:
      print("Since the rms is not available, will use LINEAR contour levels, equispaced to the peak")
      LEVS = list(np.linspace(RMS*SNR_CUT_I,np.max(imdata['I'])))
  else:
      LEVS = [lli*np.max(imdata['I']) for lli in levs]

#  LEVS = [-LEVS[0]] + LEVS


  conti = sub.contour(xyc[1],xyc[0], imdata['I'], levels = LEVS, colors='k')

  if polUnit=='mJy/beam':
    m *= 1000.0
  #  ArrowLen /= 1000.0
    mPeak *= 1000.0
  elif polUnit=='Jy/beam':
    m *= 1.0
  else:
    raise Exception("No known polUnit!")

  if vmax==0.0:
    poli = sub.imshow(np.transpose(m),extent=Extent, origin='lower', cmap='Blues',interpolation='nearest',vmin=0.0,vmax=mPeak)
  else:
    poli = sub.imshow(np.transpose(m),extent=Extent, origin='lower', cmap='Blues',interpolation='nearest',vmax=vmax*imdata['I'][polpeak],vmin=0.0)

  if figUnit=='muas':
    prUnit = r'$\mu$as'
  else:
    prUnit = figUnit

  sub.set_xlabel('Relative RA (%s)'%prUnit)
  sub.set_ylabel('Relative Dec (%s)'%prUnit)



  if doEVPA:
   Yproj = ArrowLen*np.cos(EVPA[mask])*m[mask]/mPeak
   Xproj = ArrowLen*np.sin(EVPA[mask])*m[mask]/mPeak
   lines = []
   for i in range(0,len(XLoc)):
    if XLoc[i] in SelPix and YLoc[i] in SelPixOffset[int(XLoc[i]%dPol)]:  
      Xpos = float(XLoc[i])/NPIX*(Extent[1]-Extent[0]) + Extent[0]
      Ypos = float(YLoc[i])/NPIX*(Extent[3]-Extent[2]) + Extent[2]
      X0 = float(Xpos-Xproj[i]/2.)
      X1 = float(Xpos+Xproj[i]/2.)
      Y0 = float(Ypos-Yproj[i]/2.)
      Y1 = float(Ypos+Yproj[i]/2.)
      lines.append(((X0,Y0),(X1,Y1)))
   lines = tuple(lines)

   pollines = LineCollection(lines,colors='r',linewidths=2)

   sub.add_collection(pollines)

  #fig.suptitle((r'%s - m: %.3f - $\phi$: %.1f$^\circ$. V/I = %.2e%%')%(SNAME, Pfrac, MAXPA*180./np.pi, Vpeak/peak*100.),fontsize=25)
  if len(title)==0:
      fig.suptitle((r'%s')%SNAME,fontsize=20)
  else:
      fig.suptitle(r'%s'%str(title),fontsize=20)


  beambox = AnchoredAuxTransformBox(sub.transData, loc=3,frameon=False)
  beamplot = Ellipse((0,0), width=Rbeam[0]*Kfac[figUnit], height=Rbeam[1]*Kfac[figUnit], angle=90.-Rbeam[2],alpha=0.5)
  beamplot.set_facecolor(beamcolor)
  beambox.drawing_area.add_artist(beamplot)
  sub.add_artist(beambox)

  sub.set_xlim((Extent[0]/zoom+dRA,Extent[1]/zoom+dRA))
  sub.set_ylim((Extent[2]/zoom+dDec,Extent[3]/zoom+dDec))


  pl.savefig(figName)
  return [fig,sub,conti,poli]




def plotDterms(dttable,imname,title='Solved Dterms',DtMax = 0.0):
  """ Plot Dterms from a CASA calibration table. 
      ``dttable'': Name of the Dterms table.
      ``imname'':  Name of the output image (with desired extension).
      ``title'':   Figure title.
      ``DtMax'':   If not zero, maximum axis range of the figure.
  """



  print('This function is part of the CASA-poltools.\n    Please refer to:   Marti-Vidal et al. A&A, 646, A52')

  DTMAX = float(DtMax)

  tb.open(os.path.join(dttable,'ANTENNA'))
  an = list(tb.getcol('NAME'))
  tb.close()

  tb.open(dttable)
  CPR = tb.getcol('CPARAM')
  ERR = tb.getcol('PARAMERR')
  tb.close()


  symb = ['o','^'] #,'s','*','<','>','p']
  cols = ['r','g','b','m','c','y','k','w','orange','lime','dimgray','silver','brown']

  MS = 12


  DTs = {}
  for i,ai in enumerate(an):
    DATA = [CPR[0,0,i].real,ERR[0,0,i],CPR[0,0,i].imag,ERR[0,0,i],CPR[1,0,i].real,ERR[1,0,i],CPR[1,0,i].imag,ERR[1,0,i]]
    DTs[ai] = np.array(DATA)


  fig = pl.figure(figsize=(6,6))
  sub = fig.add_subplot(111)
  fig.subplots_adjust(right=0.97, top=0.93)
    
  ABSMAX = 0.0

  outf = open(imname+'_DT-Values.txt','w')

  for i,ant in enumerate(DTs.keys()):
 #   sy = int(i/len(cols))
    co = (ANTENNAS.index(ant))%len(cols)
 #   print '%s%s'%(symb[0],cols[co])
    sub.plot(DTs[ant][0],DTs[ant][2],markerfacecolor=cols[co], marker=symb[0], markersize=MS)
    sub.errorbar(DTs[ant][0],DTs[ant][2],3.*DTs[ant][3],3.*DTs[ant][1],ecolor='k',fmt='')


    if i==0:
      sub.plot([],[],'%sk'%symb[0],label='DR',markersize=MS)


    sub.plot(DTs[ant][4],DTs[ant][6],marker=symb[1],markerfacecolor=cols[co],markersize=MS)
    sub.errorbar(DTs[ant][4],DTs[ant][6],3.*DTs[ant][5],3.*DTs[ant][7],ecolor='k',fmt='')
    if i==0:
      sub.plot([],[],'%sk'%symb[1],label='DL',markersize=MS)

    ABSMAX = np.max([ABSMAX,np.abs(DTs[ant][0]),np.abs(DTs[ant][2]),np.abs(DTs[ant][4]),np.abs(DTs[ant][6])])

    pl.text(DTs[ant][0]+0.02*ABSMAX,DTs[ant][2]+0.02*ABSMAX,ant)
    pl.text(DTs[ant][4]+0.02*ABSMAX,DTs[ant][6]+0.02*ABSMAX,ant)

    print('%s  % .3e  % .3e  % .3e  % .3e  % .3e  % .3e  % .3e  % .3e '%(ant, DTs[ant][0],DTs[ant][1], DTs[ant][2],DTs[ant][3], DTs[ant][4],DTs[ant][5], DTs[ant][6],DTs[ant][7]),file=outf)


  pl.legend(numpoints=1)

  if DtMax == 0.0:
    pl.xlim((-1.2*ABSMAX,1.2*ABSMAX))
    pl.ylim((-1.2*ABSMAX,1.2*ABSMAX))
  else:
    pl.xlim((-DTMAX,DTMAX))
    pl.ylim((-DTMAX,DTMAX))


  pl.title(title,fontsize=20)
  os.system('rm -rf %s.DTERMS.png'%imname)
  pl.savefig(imname+'.DTERMS.png')

  outf.close()









def plotPolCovMatrix(CVfile,antnam,gamma=0.4, outname = 'CovMatrix.png',kind=1):
  """ Plot a raster image of the post-fit covariance matrix
      obtained from a run of PolSolve.
      ``CVfile'': ascii file with the covariance matrix.
      ``antnam'': list of antenna names.
      ``gamma'': the gamma factor for the image scaling.
      ``kind'': if 1, plot covariance; if 2, plot correlation.
   """



  print('This function is part of the CASA-poltools.\n    Please refer to:   Marti-Vidal et al. A&A, 646, A52')

  iff = open(CVfile)
  lines = filter(lambda x: '!' not in x, iff.readlines())
  par = []
  hand = []
  data = []
  for j,line in enumerate(lines):
    temp = line.split()  
    if 'Antenna' in line or 'Component' in line: 
      if j%2 == 0:
        i = int(temp[3][:-1])
        if temp[2]=='Component':
          par.append(-(i+1))
          hand.append('')
        else:
          par.append(i+1)  
          hand.append(temp[5])
    else:  
      data.append(list(map(float,temp)))

  data = np.array(data)
  
  fig = pl.figure(figsize=(8,8))
  sub = fig.add_subplot(111)
  fig.subplots_adjust(left=0.06, bottom=0.06,top=0.98,right=0.98)
  

  corr = np.copy(data)
  if kind==2:
    Npix = np.shape(data)[0]
    for i in range(Npix):
      for j in range(Npix):  
        if data[i,i]>0.0 and data[j,j]>0.0:  
          corr[i,j] /= np.sqrt(data[i,i]*data[j,j])   
        else:
          corr[i,j] = 0.0  

    corr[np.abs(corr)>1.0] = 1.0 
    print('Extreme correlation values:')
    print(np.max(corr), np.min(corr))

  immat = sub.imshow(np.power(np.abs(corr),gamma),interpolation='nearest',cmap='gist_gray')

  sub.set_xticks(np.array(range(0,2*len(par),2))+0.7)
  sub.set_yticks(np.array(range(0,2*len(par),2))+0.7)

  ticknam = []
  for k in range(len(par)):
    if par[k] < 0:
      ticknam.append('S%i'%(np.abs(par[k])))
    else:  
      ticknam.append((r'$%s_%s$')%(antnam[np.abs(par[k])-1],hand[k]))
  
    if k>0 and par[k]>0 and par[k-1]<0:
      width = 8
    elif k<len(par)-1 and hand[k]=='L' and hand[k-1]=='R':
      width = 5
    else:
      width = 1

    sub.plot(np.array([2*k-0.5,2*k-0.5]),np.array([0-0.5,2*len(par)-0.5]),'--r',lw=width)
    sub.plot(np.array([0-0.5,2*len(par)-0.5]),np.array([2*k-0.5,2*k-0.5]),'--r',lw=width)


  sub.set_xticklabels(ticknam)
  sub.set_yticklabels(ticknam) 

  pl.savefig(outname)








def plotPolTraces(vis = [], antennas = []): #['AA','AP','AZ','LM']):
  """ Plots (a comparison of) the closure traces of all the 
      measurement sets in the "vis" list. Exactly the same UV 
      sampling is assumed for all the measurement sets in the list.
      Only the antennas in the "antennas" list are used to 
      compute the closure traces. 
  """

  print('This function is part of the CASA-poltools.\n    Please refer to:   Marti-Vidal et al. A&A, 646, A52')

  visSymb = ['or','xg','sb','+c','dk','*m','oy']

  VISNAME = vis[0]


 # Get indices of the selected antennas:
  tb.open(os.path.join(VISNAME,'ANTENNA'))
  NAM = list(tb.getcol('NAME'))

  if len(antennas)==0:
     antennas = [str(ai) for ai in NAM]

  antennas = sorted(antennas)

 ## Sanity check:
  if len(antennas)<4:
    raise Exception("Closure trace combinations require at least 4 antennas!")



  AIDX = []
  tb.close()
  for ant in antennas:
    if ant not in NAM:
      raise Exception("Antenna %s not in dataset(s)!"%ant)
    AIDX.append(NAM.index(ant))
  AIDX = np.array(AIDX,dtype=np.int32)

# Get the set of trace baselines:
  Traces = [list(ci) for ci in combinations(AIDX,4)]
  Traces = np.array(Traces,dtype=np.int32)

  # Get visibility indices:
  tb.open(VISNAME)
  A1 = tb.getcol('ANTENNA1')
  A2 = tb.getcol('ANTENNA2')
  IDX = []
  for i in range(len(A1)):
    if A1[i] in AIDX and A2[i] in AIDX:
      IDX.append(i)
  IDX = np.array(IDX)
  tb.close()


# Get some metadata:
  tb.open(os.path.join(VISNAME,'SPECTRAL_WINDOW'))
  NU = tb.getcol('REF_FREQUENCY')
  LAMBDA = 2.99792458e8/NU
  tb.close()

  ms.open(VISNAME)
  POLS = list(ms.range('corr_names')['corr_names'])
  ms.close()

  RR = POLS.index('RR')
  RL = POLS.index('RL')
  LR = POLS.index('LR')
  LL = POLS.index('LL')


# Get selected data:
  SEL_DATA = []

  tb.open(VISNAME)
  SELA1 = np.copy(A1[IDX])
  SELA2 = np.copy(A2[IDX])
  SELTIME = np.copy(tb.getcol('TIME')[IDX])
  SELUVW = np.copy(tb.getcol('UVW')[:,IDX])
  SELSPW = np.copy(tb.getcol('DATA_DESC_ID')[IDX])
  tb.close()

  for vi in vis:
    tb.open(vi)
    SEL_DATA.append(np.average(np.copy(tb.getcol('DATA')[:,:,IDX]),axis=1))
    tb.close()


## Get integration times (and time indices):
  UT = np.unique(SELTIME)

  basmask = {}
  BASNAME = []
  for ai in range(len(AIDX)-1):
    for aj in range(ai+1,len(AIDX)):
      ani = AIDX[ai]; anj = AIDX[aj]
      basmask['%i-%i'%(ani,anj)] = np.logical_and(SELA1==ani,SELA2==anj)
      BASNAME.append('%i-%i'%(ani,anj))

  NBAS = len(BASNAME)

  BASDATA_RR = [np.zeros((NBAS,len(UT)), dtype=np.complex64) for vi in vis]
  BASDATA_RL = [np.zeros((NBAS,len(UT)), dtype=np.complex64) for vi in vis]
  BASDATA_LR = [np.zeros((NBAS,len(UT)), dtype=np.complex64) for vi in vis]
  BASDATA_LL = [np.zeros((NBAS,len(UT)), dtype=np.complex64) for vi in vis]



  for bi in range(len(BASNAME)):
    tempMask = basmask[BASNAME[bi]]
    for ti in range(len(UT)):
      VisIdx = np.where(np.logical_and(tempMask,SELTIME==UT[ti]))[0]
      if len(VisIdx)>0:
        for vi in range(len(vis)):
          BASDATA_RR[vi][bi,ti] = SEL_DATA[vi][RR,VisIdx[0]]
          BASDATA_RL[vi][bi,ti] = SEL_DATA[vi][RL,VisIdx[0]]
          BASDATA_LR[vi][bi,ti] = SEL_DATA[vi][LR,VisIdx[0]]
          BASDATA_LL[vi][bi,ti] = SEL_DATA[vi][LL,VisIdx[0]]
      else:
        for vi in range(len(vis)):
          BASDATA_RR[vi][bi,ti] = np.nan
          BASDATA_RL[vi][bi,ti] = np.nan
          BASDATA_LR[vi][bi,ti] = np.nan
          BASDATA_LL[vi][bi,ti] = np.nan


  TRACES = np.zeros((len(Traces),len(UT)), dtype=np.complex64)


  fig = pl.figure(figsize=(10,5))
  subAmp = fig.add_subplot(212)
  subPhs = fig.add_subplot(211,sharex=subAmp)
  fig.subplots_adjust(hspace=0.01,wspace=0.01)

  TITLE = fig.suptitle('TEST')

  UTplot = UT/86400.
  UTplot -= int(np.min(UTplot))
  UTplot *= 24.0


  for tri in range(len(Traces)):
    subAmp.cla()
    subPhs.cla()
    a1,a2,a3,a4 = Traces[tri]
    bAB = BASNAME.index('%i-%i'%(a1,a2)); bBC = BASNAME.index('%i-%i'%(a2,a3))
    bCD = BASNAME.index('%i-%i'%(a3,a4)); bAD = BASNAME.index('%i-%i'%(a1,a4))
    TRACES[tri,:] = 0.0
    for vi in range(len(vis)):
      RRab = BASDATA_RR[vi][bAB]; RLab = BASDATA_RL[vi][bAB]
      LRab = BASDATA_LR[vi][bAB]; LLab = BASDATA_LL[vi][bAB]
      RRbc = np.conjugate(BASDATA_RR[vi][bBC]); RLbc = np.conjugate(BASDATA_RL[vi][bBC])
      LRbc = np.conjugate(BASDATA_LR[vi][bBC]); LLbc = np.conjugate(BASDATA_LL[vi][bBC])
      RRcd = BASDATA_RR[vi][bCD]; RLcd = BASDATA_RL[vi][bCD]
      LRcd = BASDATA_LR[vi][bCD]; LLcd = BASDATA_LL[vi][bCD]
      RRad = BASDATA_RR[vi][bAD]; RLad = BASDATA_RL[vi][bAD]
      LRad = BASDATA_LR[vi][bAD]; LLad = BASDATA_LL[vi][bAD]

      TRACES[tri,:] = -((LRbc*RRab - RLab*RRbc)*LRcd + (RLab*RLbc - LLbc*RRab)*RRcd)*LLad + ((LRbc*RRab - RLab*RRbc)*LLcd + (RLab*RLbc - LLbc*RRab)*RLcd)*LRad + ((LRab*LRbc - LLab*RRbc)*LRcd - (LLbc*LRab - LLab*RLbc)*RRcd)*RLad - ((LRab*LRbc - LLab*RRbc)*LLcd - (LLbc*LRab - LLab*RLbc)*RLcd)*RRad

      TRACES[tri,:] /= 2.*(LRad*RLad - LLad*RRad)*(LRbc*RLbc - LLbc*RRbc)

      subAmp.plot(UTplot,np.abs(TRACES[tri,:]),visSymb[vi],mew=2)
      subPhs.plot(UTplot,180./np.pi*np.angle(TRACES[tri,:]),visSymb[vi],label=os.path.basename(vis[vi]),mew=2)

    subPhs.set_ylim((-180.,180.))
    subAmp.plot(np.array([UTplot[0],UTplot[-1]]),np.array([1.,1.]),':k')
    subPhs.plot(np.array([UTplot[0],UTplot[-1]]),np.array([0.,0.]),':k')
    pl.setp(subPhs.get_xticklabels(),'visible',False)
    pl.legend(numpoints=1,ncol=3,prop={'size':10})
    subAmp.set_xlabel('UT (h)')
    subAmp.set_ylabel(r'$|\mathcal{T}|$')
    subPhs.set_ylabel(r'$\mathrm{arg}(\mathcal{T}) (deg)$')
    TITLE.set_text('%s-%s-%s-%s'%(NAM[a1],NAM[a2],NAM[a3],NAM[a4]))

    pl.savefig('TRACE_PLOT_%s-%s-%s-%s.png'%(NAM[a1],NAM[a2],NAM[a3],NAM[a4]))   



