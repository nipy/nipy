#!/usr/bin/env python
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Build helper."""

__docformat__ = 'restructuredtext'

from os.path import join

# Debug import
"""
import neuroimaging.externals.pynifti # directory /Users/cburns/src/nipy/neuroimaging/externals/pynifti

# /Users/cburns/src/nipy/neuroimaging/externals/pynifti/__init__.pyc matches /Users/cburns/src/nipy/neuroimaging/externals/pynifti/__init__.py

import neuroimaging.externals.pynifti # precompiled from /Users/cburns/src/nipy/neuroimaging/externals/pynifti/__init__.pyc

import neuroimaging.externals.pynifti.nifti # directory /Users/cburns/src/nipy/neuroimaging/externals/pynifti/nifti

# /Users/cburns/src/nipy/neuroimaging/externals/pynifti/nifti/__init__.pyc matches /Users/cburns/src/nipy/neuroimaging/externals/pynifti/nifti/__init__.py

import neuroimaging.externals.pynifti.nifti # precompiled from /Users/cburns/src/nipy/neuroimaging/externals/pynifti/nifti/__init__.pyc

import nifti # directory /Users/cburns/local/lib/python2.5/site-packages/nifti

# /Users/cburns/local/lib/python2.5/site-packages/nifti/__init__.pyc matches /Users/cburns/local/lib/python2.5/site-packages/nifti/__init__.py

import nifti # precompiled from /Users/cburns/local/lib/python2.5/site-packages/nifti/__init__.pyc

# /Users/cburns/local/lib/python2.5/site-packages/nifti/niftiimage.pyc matches /Users/cburns/local/lib/python2.5/site-packages/nifti/niftiimage.py

import nifti.niftiimage # precompiled from /Users/cburns/local/lib/python2.5/site-packages/nifti/niftiimage.pyc

# /Users/cburns/local/lib/python2.5/site-packages/nifti/nifticlib.pyc matches /Users/cburns/local/lib/python2.5/site-packages/nifti/nifticlib.py

import nifti.nifticlib # precompiled from /Users/cburns/local/lib/python2.5/site-packages/nifti/nifticlib.pyc

dlopen("/Users/cburns/local/lib/python2.5/site-packages/nifti/_nifticlib.so", 2);

import nifti._nifticlib # dynamically loaded from /Users/cburns/local/lib/python2.5/site-packages/nifti/_nifticlib.so

"""

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    # debug
    from numpy.distutils import system_info
    system_info.verbosity = 1

    config = Configuration('nifti', parent_package, top_path)
 
    nifticlib_include_dirs = [get_numpy_include_dirs(),
                              join('nifti', 'nifticlib', 'include')]

    # znz library
    znzlib_src = join('nifti', 'nifticlib', 'znzlib', 'znzlib.c')
    config.add_library('znz',
                       sources = znzlib_src,
                       headers = join('nifti', 'nifticlib', 'znzlib', 
                                      'znzlib.h'),
                       libraries = 'z')

    # niftiio library
    niftiio_src = join('nifti', 'nifticlib', 'niftilib', 'nifti1_io.c')
    config.add_library('niftilib',
                       sources = niftiio_src,
                       headers = join('nifti', 'nifticlib', 'niftilib',
                                    'nifti1_io.h'),
                       include_dirs = nifticlib_include_dirs)

    # nifticlib extension
    nifticlib_src = join('nifti', 'nifticlib_wrap.c')
    config.add_extension(join('nifti', '_nifticlib'),
    #config.add_extension('_nifticlib',
                         sources = nifticlib_src,
                         include_dirs = [nifticlib_include_dirs,
                                         join('usr', 'lib')],
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
