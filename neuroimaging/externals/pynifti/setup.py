#!/usr/bin/env python

"""
Build script for pynifti and nifticlibs.  Modified form the original version
that required SWIG for building, include the generated swig wrapper file.

External Dependency
-------------------
The nifticlibs depends on the zlib library.  This should be installed on all
Unix based systems, it may require an extra install on Windows.
http://www.zlib.net/

#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
"""

__docformat__ = 'restructuredtext'

from os.path import join

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('nifti', parent_package, top_path)
 
    nifticlib_include_dirs = [get_numpy_include_dirs(),
                              join('nifti', 'nifticlib', 'include')]

    # The c source files have several ifdef's that only compile in
    # support for zlib if this is defined.
    define_have_zlib = ('HAVE_ZLIB', '1')

    # znz library
    znzlib_src = join('nifti', 'nifticlib', 'znzlib', 'znzlib.c')
    config.add_library('znz',
                       sources = znzlib_src,
                       macros = [define_have_zlib],
                       headers = join('nifti', 'nifticlib', 'znzlib', 
                                      'znzlib.h'),
                       libraries = 'z')

    # niftiio library
    niftiio_src = join('nifti', 'nifticlib', 'niftilib', 'nifti1_io.c')
    config.add_library('niftilib',
                       sources = niftiio_src,
                       macros = [define_have_zlib],
                       headers = join('nifti', 'nifticlib', 'niftilib',
                                    'nifti1_io.h'),
                       include_dirs = nifticlib_include_dirs)

    # nifticlib extension
    nifticlib_src = join('nifti', 'nifticlib_wrap.c')
    config.add_extension(join('nifti', '_nifticlib'),
                         sources = nifticlib_src,
                         define_macros = [define_have_zlib],
                         include_dirs = [nifticlib_include_dirs],
                         libraries = ['znz', 'niftilib'],
                         depends = [znzlib_src, niftiio_src])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
    """
    setup(version      = '0.20070930.1',
          author       = 'Michael Hanke',
          author_email = 'michael.hanke@gmail.com',
          license      = 'MIT License',
          url          = 'http://apsy.gse.uni-magdeburg.de/hanke',
          description  = 'Python interface for the NIfTI IO libraries',
          **configuration(top_path='').todict())
    """
