#!/usr/bin/env python
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Build helper."""

__docformat__ = 'restructuredtext'

#from distutils.core import setup, Extension
#import os
from os.path import join
#import numpy as N
#from glob import glob

# We don't want to require swig... include generated C files
"""
nifti_wrapper_file = os.path.join('nifti', 'nifticlib.py')

# create an empty file to workaround crappy swig wrapper installation
if not os.path.isfile(nifti_wrapper_file):
    open(nifti_wrapper_file, 'w')
"""

# find numpy headers
#numpy_headers = os.path.join(os.path.dirname(N.__file__),'core','include')


# Notes on the setup
# Version scheme is:
# 0.<4-digit-year><2-digit-month><2-digit-day>.<ever-increasing-integer>
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('nifti', parent_package, top_path)
 
    nifticlib_include_dirs = [get_numpy_include_dirs(),
                              join('nifti', 'nifticlib', 'include')]

    znzlib_src = join('nifti', 'nifticlib', 'znzlib', 'znzlib.c')
    config.add_library('znz',
                       sources=znzlib_src,
                       headers=join('nifti', 'nifticlib', 'znzlib', 'znzlib.h'))

    niftiio_src = join('nifti', 'nifticlib', 'niftilib', 'nifti1_io.c')
    config.add_library('niftilib',
                       sources=niftiio_src,
                       headers=join('nifti', 'nifticlib', 'niftilib',
                                    'nifti1_io.h'),
                       include_dirs = nifticlib_include_dirs)

    #config.add_extension(join('nifti', '_nifticlib'),
    config.add_extension('_nifticlib',
                         sources = [join('nifti', 'nifticlib_wrap.c')],
                         include_dirs = nifticlib_include_dirs,
                         libraries = ['znz', 'niftilib'],
                         depends = [znzlib_src, niftiio_src])

    #                     library_dirs = [join('nifti', 'nifticlib', 'znzlib')],

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

          
"""
setup(name       = 'pynifti',
    version      = '0.20070930.1',
    author       = 'Michael Hanke',
    author_email = 'michael.hanke@gmail.com',
    license      = 'MIT License',
    url          = 'http://apsy.gse.uni-magdeburg.de/hanke',
    description  = 'Python interface for the NIfTI IO libraries',
    long_description = """ """,
    packages     = [ 'nifti' ],
    scripts      = glob( 'bin/*' ),
    ext_modules  = [ Extension( 'nifti._nifticlib', [ 'nifti/nifticlib.i' ],
            include_dirs = [ '/usr/include/nifti', numpy_headers ],
            libraries    = [ 'niftiio', 'znz', 'z' ],
            swig_opts    = [ '-I/usr/include/nifti',
                             '-I' + numpy_headers ] ) ]
    )
"""
