from distutils.core import setup, Extension
import numpy as np
import os



printM  = '\n'
printM += '#######################################################################\n'
printM += '# Compiling with numpy version %s \n'%(np.__version__)
printM += '#                              %s\n'%(np.__file__)
printM += '#######################################################################\n'
printM += '\n'

print(printM)




sourcefiles = ['PolSolver.cpp']

c_ext = Extension("PolSolver", sources=sourcefiles,
                  extra_compile_args=["-Wno-deprecated","-O3"],
                  include_dirs=[np.get_include()])
             #     libraries=['cfitsio','gsl','fftw3'],
             #     extra_link_args=["-Xlinker"], "-export-dynamic"])

setup(ext_modules=[c_ext],include_dirs=['./'])




