# CCextract - A task to prepare CC files for PolSolve.
#
# Copyright (c) Ivan Marti-Vidal - Universitat de Valencia (2020).
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

import gc
import numpy as np
import datetime as dt
import os,sys

from casatools import imager #as im
from casatools import image #as ia
from casatools import regionmanager #as rg

im = imager()
ia = image()
rg = regionmanager()


__version__ = '1.9b'

__help__ = """

Extracts CC components from CASA model images, using CASA regions.

Please, cite this software as: 

             Marti-Vidal et al. 2021, A&A, 646, 52


PARAMETERS:

   model_image :: CASA model image.

   regions :: List of regions (one per CC subcomponent). Follows the CASA region format (e.g. "circle[[64pix,64pix],16pix]"). Default (empty list) means to take all CLEAN components as one (sub)component. Also supports a region file in DS9 format (in pixel units). If set to "manual", a graphical interface will open to generate the regions manually.

   make_plot :: If True, a figure will be created with the source image (in greyscale) and the locations of the CLEAN models (different sub-components in different colors).


USAGE:

After running (t)clean, the CASA images (with the '.image' extension) do not have CC component tables. This task generates FITS image files with CC tables in AIPS format, separated in subcomponents that can be specified in CASA region format. In addition to this, it creates ascii files with the CC information of each sub-component, which can be used in PolSolve. 



"""

#####################
# UNIT TEST LINES:
if __name__=='__main__':

  model_image='Ex1_v0.model'
  regions=['circle[[256pix,256pix],13pix]',
                   'circle[[226pix,286pix],13pix]']
  make_plot=True

  model_image         = 'TOTO2.image'
  regions             = ['rotbox[[486pix,486pix],[26pix,14pix],-45deg]', 
                         'rotbox[[491pix,496pix],[25pix,13pix],-45deg]', 
                         'rotbox[[498pix,505pix],[25pix,14pix],-45deg]', 
                         'rotbox[[521pix,530pix],[25pix,13pix],-45deg]', 
                         'rotbox[[527pix,537pix],[24pix,13pix],-45deg]'] 



  model_image        =  "SgrA_clean.model"
  regions            =  "manual"
  make_plot          =  True



#
#
##################






def CCextract(model_image = '',regions=[],make_plot=False):

  """ 
     
      Program CCextract by I. Marti-Vidal (Univ. Valencia).

      Execute CCextract() to get some help text. 
  """

  import pyfits as pf
  try:
    import dissect_cleancomp
    isTk = True
  except:
    print("\n\n    Your CASA does not seem to have Tkinter working.\n    You cannot use the GUI in CCextract")
    isTk = False

########
# Uncomment for testing/debugging
#  return True
#if __name__=='__main__':
########


  DEBUG = False

  def printError(msg,logfile=""):
    printMsg('\n %s \n'%msg)
    raise Exception(msg)

  def printMsg(msg,logfile="",dolog=True):
    print(msg)
    if dolog and len(logfile)>0:
       off = open(logfile,'a')
       tstamp = dt.datetime.now().strftime("%Y-%M-%d/%H:%M:%S")
       print('%s: %s'%(tstamp,msg),file=off)
       off.close()



  units = {'rad':180./np.pi, 'deg': 1.0, 'arcsec': 1./3600.}
  colors = ['r','g','b','c','y','m','k']


  printMsg('\n\n  CCEXTRACT - VERSION %s  - I. Marti-Vidal (Universitat de Valencia, Spain)'%__version__)

  if len(model_image)==0:
     printMsg(__help__,dolog=False)
     return True



  LOGNAME = '%s_CCextract.log'%os.path.basename(model_image)
  if os.path.exists(LOGNAME):
    os.system('rm -rf %s'%LOGNAME)


  if not os.path.exists(model_image):
    printError('ERROR: model_image does not exist!',LOGNAME)

  try:
    ia.open(model_image)
  except:
    printError('ERROR: model_image is not a valid CADA model image',LOGNAME)


  if ia.summary()['unit'] != 'Jy/pixel':
    printMsg('WARNING: Image units do not agree with a DECONVOLVED model!',LOGNAME)


  X0,Y0 = ia.summary()['refpix'][:2]
  DX,DY = ia.summary()['incr'][:2]
  NX,NY = ia.summary()['shape'][:2]

  DX *= units[ia.summary()['axisunits'][0]]
  DY *= units[ia.summary()['axisunits'][1]]

  RAnge = [-X0*DX,(NX-X0)*DX,-Y0*DY,(NY-Y0)*DY]

  csys = ia.coordsys()
  rg.setcoordinates(csys.torecord())

  os.system('rm -rf CCextract.temp')
  os.system('cp -r %s CCextract.temp'%model_image)
  ia.close()

  ia.open('CCextract.temp')
  MM = ia.getchunk()
  mask = np.copy(MM)
  mask[:] = 0.0
  ia.putchunk(mask)
  ia.close()


  printMsg('Model file has a total of %.5f Jy'%np.sum(MM),LOGNAME)


  CASAReg = []

  if type(regions) is str and regions!='manual':
    if os.path.exists(regions):
      IFF = open(regions)
      allLines = IFF.readlines()
      IFF.close()

      ## DS9 file:
      if ' DS9 ' in allLines[0]:
        for line in allLines[1:]:
          if line.startswith('box'):
            temp = [hdp for hdp in map(float,(line.replace('box(','').replace(')','')).split(','))]
            CASAReg.append('box[[%.2fpix,%.2fpix],[%.2fpix,%.2fpix]]'%(temp[0]-temp[2]/2.,temp[1]-temp[3]/2.,temp[0]+temp[2]/2.,temp[1]+temp[3]/2.))
          if line.startswith('ellipse'):
            temp = [hdp for hdp in map(float,(line.replace('box(','').replace(')','')).split(','))]
            CASAReg.append('ellipse[[%.2fpix,%.2fpix],[%.2fpix,%.2fpix],%.2fdeg]'%(temp[0],temp[1],temp[2],temp[3],temp[4]))  
      # Alternative is CASA format:
      elif '#CRTF' in allLines[0]:
        for line in allLines[1:]:
          if line.startswith('box') or line.startswith('ellipse'):
            CASAReg.append(line) 
      ## Otherwise, bad file:
      else:
        raise Exception("ERROR! Bad Region file!")

    else:
      raise Exception("ERROR: Region File not found!")

  elif regions=='manual':
    printMsg('Selected option to draw regions manually.',LOGNAME)
    CASAReg = ['box[[0pix,0pix],[%ipix,%ipix]]'%(NX-1,NY-1)]

  elif len(regions)==0:
     printMsg('No region specifield. Will take the whole image',LOGNAME)
     CASAReg = ['box[[0pix,0pix],[%ipix,%ipix]]'%(NX-1,NY-1)]

  else:
     for reg in regions:
       CASAReg.append(str(reg))



  CCs = []  

  TotFlux = 0.0

  imname = '.'.join(model_image.split('.')[:-1])
  isFITS = False
  if os.path.exists(imname+'.image'):
    isFITS = True
    printMsg('CASA convolved image found! Will export to FITS and add the CC tables',LOGNAME)
    ia.open(imname+'.image')
    os.system('rm -rf %s_orig.fits'%imname)
    os.system('rm -rf %s.fits'%imname)
    ia.tofits(imname+'_orig.fits')
    ia.close()
    ffile = pf.open(imname+'_orig.fits')


  tbhdu = []

  for i,ri in enumerate(CASAReg):
    ia.open('CCextract.temp')
    temp = ia.getchunk()
    temp[:] = 0.0
    ia.putchunk(temp)
    shi = ia.summary()['shape']
    csys = ia.coordsys()
    ia.close()

    rg.setcoordinates(csys.torecord())
    r1 = rg.fromtext(ri,shape=shi)

    im.regiontoimagemask('CCextract.temp', r1)
    ia.open('CCextract.temp')
    mask = np.where(ia.getchunk()[:,:,0,0]>0.0)
    ia.close()
    OFF = open('%s.CC%02i'%(model_image,i),'w')

    FLUX = [] ; RA = [] ; DEC = [] ; AUX = []

    for pix in range(len(mask[0])):
      if MM[mask[0][pix],mask[1][pix],0,0]!=0.0:
        IJpix = mask[0][pix]*NY + mask[1][pix]
        if IJpix in CCs:
          printMsg('WARNING: pixel [%i,%i] already taken! Will not include it in the %ith region'%(mask[0][pix],mask[1][pix],i+1),LOGNAME)
        else:
          CCs.append(IJpix)
          FLUX.append(MM[mask[0][pix],mask[1][pix],0,0])
          RA.append((mask[0][pix]-X0)*DX)
          DEC.append((mask[1][pix]-Y0)*DY)
          TotFlux += FLUX[-1]
          print('%i  %.3e %.4e  %.4e'%(pix,FLUX[-1],RA[-1],DEC[-1]),file=OFF)

    if isFITS:
      col1 = pf.Column(name='FLUX',format='1E',array=FLUX,unit='JY')
      col2 = pf.Column(name='DELTAX',format='1E',array=RA,unit='DEGREES')
      col3 = pf.Column(name='DELTAY',format='1E',array=DEC,unit='DEGREES')
      col4 = pf.Column(name='MAJOR AX',format='1E',array=AUX,unit='DEGREES')
      col5 = pf.Column(name='MINOR AX',format='1E',array=AUX,unit='DEGREES')
      col6 = pf.Column(name='POSANGLE',format='1E',array=AUX,unit='DEGREES')
      col7 = pf.Column(name='TYPE OBJ',format='1E',array=AUX,unit='CODE')
      cols = pf.ColDefs([col1,col2,col3,col4,col5,col6,col7])
      tbhdu.append(pf.BinTableHDU.from_columns(cols))
      tbhdu[-1].header['EXTNAME'] = 'AIPS CC'

    OFF.close()
    if len(FLUX)==0:
      os.system('rm -rf %s.CC%02i'%(model_image,i))


  if isFITS:
    thdulist = pf.HDUList([ffile[0]] + tbhdu)
    thdulist.writeto('%s.fits'%imname)
    ffile.close()
    os.system('rm -rf %s_orig.fits'%imname)

    ia.open(imname+'.image')
    IMG = ia.getchunk()
    ia.close()

    IM2PLT = [np.transpose(np.log(IMG[:,:,0,0]-np.min(IMG)+0.0001)),RAnge]
  else: 
    IM2PLT = None

  printMsg('Total flux within the selected regions: %.5f Jy'%TotFlux,LOGNAME)
  fname0 = '{0}.CC00'.format(model_image)
  if regions=='manual' and os.path.isfile(fname0):
    if isTk:
       fname1 = '{0}.CC'.format(model_image)
       os.rename(fname0, fname1)
       CASAReg = dissect_cleancomp.dissect_cleancomp(fname1, 'X-F-RA_DEC', plotsize=32, savetodisk='%02i',image=IM2PLT)
       os.remove(fname1)
    else:
       printError("You cannot run in manual mode, since tkInter is not working in this CASA",LOGNAME)

  os.system('rm -rf CCextract.temp')


  if isFITS and make_plot:
    import pylab as pl
    NX,NY = np.shape(IMG)[:2]
    fig = pl.figure()
    sub = fig.add_subplot(111)
    sub.imshow(IM2PLT[0], origin='lower', cmap='Greys')
    sub.set_xlabel('RA pixel')
    sub.set_ylabel('Dec pixel')
    fig.suptitle(os.path.basename(imname),fontsize=25)
    
    for i,ri in enumerate(CASAReg):
      if os.path.exists('%s.CC%02i'%(model_image,i)):  
        OFF = open('%s.CC%02i'%(model_image,i))
        ci = i%len(colors)
        Xpix = [] ; Ypix = []
        for line in OFF.readlines():
          temp = [hdp for hdp in map(float,line.split())][1:]
          Xpix.append(int(temp[1]/DX + X0))
          Ypix.append(int(temp[2]/DY + Y0))
        Xpix = np.array(Xpix) ; Ypix = np.array(Ypix)
        sub.plot(Xpix,Ypix,'x%s'%colors[ci])
      sub.set_xlim((0,NX)) ; sub.set_ylim((0,NY))
      pl.savefig('%s_CCplot.png'%imname)


  print("DONE!")
