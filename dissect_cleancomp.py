#!/usr/bin/env python
#
# Copyright (C) 2019 Michael Janssen
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Library General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public
# License for more details.
#
# You should have received a copy of the GNU Library General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 675 Massachusetts Ave, Cambridge, MA 02139, USA.
#




###############
## Nov 9, 2020: Small edits to keep the plot open over sub-components (I. Marti-Vidal).
###############


"""
Use an interactive plotting window to dissect a CLEAN component map into multiple sub-components.
This is useful for an LPCAL-like interferometry polarization leakage calibration, where D-terms need to be solved
on compact source sub-components.
The main functon is dissect_cleancomp(components_file). Its docstring contains a description of the dissection process.
"""

import sys
from optparse import OptionParser
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
import _tkinter
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def main():
    """
    Calls dissect_cleancomp with a components_file from the command line.
    """
    usage  = "%prog CLEAN-components-file [-f fileformat] [-h or --help]"
    parser = OptionParser(usage=usage)
    parser.add_option("-f", type="string",dest="filefmt",default="F-R-T",
                      help=r"Format of CLEAN-components-file. [Default: %default]")
    (opts,args) = parser.parse_args()
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    dissect_cleancomp(args[0], opts.filefmt)








def dissect_cleancomp(components_file, filefmt='F-R-T', fontsize=18, plotsize=None, savetodisk=False,image=None):
    """
    Takes an ASCII CLEAN components_file and splits it into multiple sub-components.
    The components_file must follow a format specified by the filefmt input.
    If filefmt=='RA-DEC-F':
        #RAoffset[arcsec], Decoffset[arcsec], Flux[Jy]
        0.01, 0.0, 0.36
        0.42, 1.2, 0.01
        #...
    If filefmt=='X-F-RA_DEC':
        #skipped, Flux[Jy], RAoffset[rad], Decoffset[rad]
        360623  1.174e-03 9.7222e-08  2.2222e-08
        362720  1.182e-04 8.8889e-08  2.2222e-08
        363419  1.475e-04 8.6111e-08  2.2222e-08
        #...
    If filefmt=='F-R-T':
        !Flux (Jy) Radius (mas)  Theta (deg)
        0.162522   0.00500000    90.0000
        0.0258326  0.00707107    135.000
        0.128304   0.00500000    180.000
        !...
    If filefmt=='meqsil': !will be converted to mas scale!
        #name ra_h ra_m ra_s dec_d dec_m dec_s i q u v emaj_s emin_s pa_d
        s0 -0.0 0.0 2.66666666667e-07 0.0 0.0 2.44929359829e-22 0.0960342 0 0 1e-16 1e-16 0.0
        s1 -0.0 0.0 2.66667026958e-07 -0.0 0.0 1.99999823614e-06 0.0217345 0 0 1e-16 1e-16 0.0
        s2 -0.0 0.0 1.33333468874e-07 -0.0 0.0 2.00000203311e-06 0.0922213 0 0 1e-16 1e-16 0.0
        s3 -0.0 0.0 4e-07 0.0 0.0 3.67394039744e-22 0.0739824 0 0 1e-16 1e-16 0.0
        #...
    An interactive GUI will open where the user can draw polygons to select components.
    Components with negative fluxes are drawn as black crosses.

    Returns a dict that contains (RAoffset, Decoffset, flux) values in units given by the input file, for each group of
    components inside a polygon.
    If savetodisk is set to True, CLEAN component files will be written to disk. If N polygons are drawn, N files will be
    written, called '<components_file>.i' for i=0,1,2,...,N.
    Similarly, savetodisk can also be a string (e.g., .CC%02i) to save files with a specified formatting
    (<components_file>.CC%02i%i).

    Can control the font sizes and sizes of the plotted data points with the fontsize and plotsize arguments, respectively.
    """
    raoff, decoff, flux, compnum = read_cleancomp(components_file, filefmt)
    return plotandsave(raoff, decoff, flux, compnum, components_file, fontsize, plotsize, savetodisk,image)


def read_cleancomp(_components_file, filefmt):
    """
    Reads an ASCII text file with the format specified in dissect_cleancomp().
    """
    if filefmt == 'meqsil':
        from astropy import units as u
        from astropy.coordinates import Angle
    _x  = []
    _y  = []
    _z  = []
    _N  = []
    _x0 = False
    _y0 = False
    with open(_components_file, 'r') as cf:
        for line in proper_line(cf):
            if ',' in line:
                xyz = line.split(',')
            else:
                xyz = line.split()
            if filefmt == 'RA-DEC-F':
                _x.append(float(xyz[0]))
                _y.append(float(xyz[1]))
                _z.append(float(xyz[2]))
            elif filefmt == 'X-F-RA_DEC':
                _N.append(int(xyz[0]))
                _z.append(float(xyz[1]))
                _x.append(float(xyz[2]) * 206264806.2471)
                _y.append(float(xyz[3]) * 206264806.2471)
            elif filefmt == 'F-R-T':
                _z.append(float(xyz[0]))
                _x.append(float(xyz[1])*np.sin(float(xyz[2])*np.pi/180.))
                _y.append(float(xyz[1])*np.cos(float(xyz[2])*np.pi/180.))
            elif filefmt == 'meqsil':
                rh = format(float(xyz[1]), 'f').rstrip('0').rstrip('.')
                rm = format(float(xyz[2]), 'f').rstrip('0').rstrip('.')
                rs = format(float(xyz[3]), '.16f')
                dd = format(float(xyz[4]), 'f').rstrip('0').rstrip('.')
                dm = format(float(xyz[5]), 'f').rstrip('0').rstrip('.')
                ds = format(float(xyz[6]), '.16f')
                if isinstance(_x0, bool):
                    _x0 = Angle('{0}h{1}m{2}s'.format(rh, rm, rs), unit=u.mas).value
                if isinstance(_y0, bool):
                    _y0 = Angle('{0}d{1}m{2}s'.format(dd, dm, ds), unit=u.mas).value
                _x.append(Angle('{0}h{1}m{2}s'.format(rh, rm, rs), unit=u.mas).value - _x0)
                _y.append(Angle('{0}d{1}m{2}s'.format(dd, dm, ds), unit=u.mas).value - _y0)
                _z.append(float(xyz[7]))
            else:
                raise ValueError('{0} is not a valid file format. Must be RA-DEC-F, X-F-RA_DEC, F-R-T, or meqsil'.format(filefmt))
    return np.asarray(_x), np.asarray(_y), np.asarray(_z), np.asarray(_N)


def proper_line(f, comment_chars = ['#', '!']):
    """
    Returns only non-commented and non-blank lines from input file.
    Lets lines continue if they end with a \ char.
    """
    lline = None
    for l in f:
        lline = l.rstrip()
        lline = lline.rstrip('\n')
        if lline:
            if lline[0] not in comment_chars:
                while lline.endswith('\\'):
                    lline = lline[:-1] + next(f).rstrip().rstrip('\n')
                yield lline


def savetofile(this_x, this_y, this_z, this_N, outfile_name):
    """
    Write dissected pieces for plotandsave().
    """
    outf = open(outfile_name, 'w')
    for _x, _y, _z, _N in zip(this_x, this_y, this_z, this_N):
        if len(this_N) == 0:
            outf.write('{0},{1},{2}\n'.format(str(_x/206264806.2471), str(_y/206264806.2471), str(_z)))
        else:
            outf.write('%i  %.3e %.4e  %.4e\n'%(_N,_z,_x/206264806.2471,_y/206264806.2471))
    outf.close()


def makeplot(x, y, z, fontsize, plotsize):
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111,aspect=1)
    fig.subplots_adjust(bottom=0.05,top=0.8,left=0.05)
    plt.setp(plt.gca()) #, autoscale_on=True)

## NOTE: plotsize=None is the default value of scatter. No harm to set it to None.
    Extreme = np.max(np.abs(z)) 
    

    posplot = plt.scatter(x, y, c=z, cmap='jet', vmin=-Extreme, vmax=Extreme, #norm=colors.PowerNorm(gamma=0.5), #norm=colors.SymLogNorm(vmin=-Extreme, vmax=Extreme),
                          s=plotsize)
    cbar = plt.colorbar(posplot, pad=0)
    cbar.set_label('Flux', fontsize = fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.gca().invert_xaxis()
    ax.set_xlabel('Right Ascension Offset', fontsize = fontsize)
    ax.set_ylabel('Declination Offset', fontsize = fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    mng = plt.get_current_fig_manager()
    return ax, posplot



def if_inside_polygon(point_x, point_y, polygon_points, dummy_sigma=''):
    point    = (point_x, point_y)
    _polygon = mpath.Path(polygon_points)
    return _polygon.contains_point(point)


def _tellme(s, fontsize):
    plt.title(s, fontsize=fontsize)
    plt.draw()


def _draw_mask(x, y, z, fontsize, plotsize,myplot=0,Iam=0):

    selcol = ['r','g','b','c','y','m','k'][Iam%7]

    if myplot==0:
      myplot = makeplot(x, y, z, fontsize, plotsize)
      
    Xaux = np.copy(x); Yaux = np.copy(y); Zaux=np.copy(z) 
    OffAux = np.array([[x[i],y[i]] for i in range(len(x))])
    myplot[1].set_offsets(OffAux)
    myplot[1].set_array(Zaux)
    plt.draw()


    plt.sca(myplot[0])
    
    happy = False
    while not happy:
        pts = []
        _tellme('Select the corners of a polygon with left mouse clicks.\n'
                'Press Backspace to remove the latest corner drawn.\n'
                'Press Enter when all corners are drawn.\n'
                'Close the plot to exit with the current selection.',
                fontsize
               )
        pts = np.asarray(plt.ginput(-1, timeout=-1,mouse_pop=None,mouse_stop=None))
        if pts.any():
            ph = plt.fill(pts[:, 0], pts[:, 1], selcol, lw=2,alpha=0.5)
            _tellme('Happy? Hit any key for yes; click the mouse to start over again.', fontsize)
            happy = plt.waitforbuttonpress()
        else:
            happy = True
        # Get rid of fill
        if not happy:
            for p in ph:
                p.remove()
  #  plt.close()
    return pts


def _maskit(x, y, z, N, fontsize, plotsize,myplot=0,Iam=0):
    try:
        mask_area = _draw_mask(x, y, z, fontsize, plotsize,myplot,Iam)
    except _tkinter.TclError:
        print('Done: Plot closed.')
        return False
    thiscomp_x = []
    thiscomp_y = []
    thiscomp_z = []
    thiscomp_N = []
    if mask_area.any():
        mask = np.zeros(len(x),dtype=bool)
        for i,xval in enumerate(x):
            yval   = y[i]
            inside = if_inside_polygon(xval,yval, mask_area)
            mask[i] = not inside

        thiscomp_x = np.copy(x[np.logical_not(mask)])
        thiscomp_y = np.copy(y[np.logical_not(mask)])
        thiscomp_z = np.copy(z[np.logical_not(mask)])
        thiscomp_N = np.copy(N[np.logical_not(mask)])
        _x = np.copy(x[mask])   #np.ma.masked_array(x, mask))
        _y = np.copy(y[mask])   #np.ma.masked_array(y, mask))
        _z = np.copy(z[mask])   #np.ma.masked_array(z, mask))
        _N = np.copy(N[mask])   #np.ma.masked_array(N, mask))
    else:
        _x, _y, _z, _N = x, y, z, N
    return _x, _y, _z, _N, thiscomp_x, thiscomp_y, thiscomp_z, thiscomp_N


def plotandsave(x,y,z,N, outfile0, fontsize, plotsize, savetodisk,image):
    """
     - Make plots of (x,y) on a grid, color-coding with z as intensity (flux) values.
     - User can draw a polygon in the plot with left mouse clicks.
     - Can be done iteratively.
     - Saves points inside polygon in a new file for each iteration, following the description of dissect_cleancomp().
    """
    gotall        = False
    iteration     = 0
    subcomponents = {}

    myplot = makeplot(x, y, z, fontsize, plotsize)      
    plt.sca(myplot[0])

    if image is not None:
       plt.imshow(image[0],origin='lower',extent=[d*206264806.2471 for d in image[1]],cmap='Greys')

    XM = np.max(np.concatenate([np.abs(x),np.abs(y)]))*1.1
    
    myplot[0].set_xlim((XM,-XM))
    myplot[0].set_ylim((-XM,XM))


    allOutFiles = []
    while not gotall:
        _go_on = _maskit(x, y, z, N, fontsize, plotsize,myplot,iteration)
        if not _go_on:
            return subcomponents
        x, y, z, N, comp_ix, comp_iy, comp_iz, comp_iN = _go_on
        if len(comp_ix)>0:
            F_tot  = str(sum(comp_iz))
            N_comp = str(len(comp_ix))
            print('>> Got {0} components with a total flux of {1} in region number {2}.'.format(N_comp, F_tot, str(iteration)))
            if savetodisk:
                if isinstance(savetodisk, str):
                    this_outf = outfile0+savetodisk%(iteration)
                else:
                    this_outf = outfile0+'.'+str(iteration)
                savetofile(comp_ix, comp_iy, comp_iz, comp_iN, this_outf)
                allOutFiles.append(this_outf)
                print('   Wrote components to {0}.'.format(this_outf))
            subcomponents[iteration] = (comp_ix, comp_iy, comp_iz, comp_iN)
            iteration += 1
        if not any(x):
            print('Done: All source components are sub-divided.')
            gotall = True

    plt.close()
    return allOutFiles


if __name__=="__main__":
    main()
