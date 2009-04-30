#!/usr/bin/env python
"""Copy source files from pynifti git directory into nipy source directory.
We only want to copy the files necessary to build pynifti and the nifticlibs,
and use them within nipy.  We will not copy docs, tests, etc...

Pynifti should be build before this script is run so swig generates the 
wrapper for nifticlib.  We do not want swig as a dependency for nipy.

"""

from os import mkdir
from os.path import join, exists, expanduser
from shutil import copy2 as copy

"""
The pynifti source should be in a directory level with nipy-trunk
Ex: 
    /Users/cburns/src/nipy
    /Users/cburns/src/pynifti
"""

src_dir = expanduser('~/src/pynifti')

# Destination directory is the top-level externals/pynifti directory
dst_dir = '..'

assert exists(src_dir)

copy(join(src_dir, 'AUTHOR'), join(dst_dir, 'AUTHOR'))
copy(join(src_dir, 'COPYING'), join(dst_dir, 'COPYING'))

# pynifti source and swig wrappers
nifti_list = ['niftiformat.py', 'niftiimage.py', 'utils.py',
              'nifticlib.py', 'nifticlib_wrap.c']
nifti_src = join(src_dir, 'nifti')
nifti_dst = join(dst_dir, 'nifti')
if not exists(nifti_dst):
    mkdir(nifti_dst)

def copynifti(filename):
    copy(join(nifti_src, filename), join(nifti_dst, filename))

for nf in nifti_list:
    copynifti(nf)

# nifticlib sources
nifticlib_list = ['LICENSE', 'README', 'nifti1.h', 'nifti1_io.c',
                  'nifti1_io.h', 'znzlib.c', 'znzlib.h'] 
nifticlib_src = join(src_dir, '3rd', 'nifticlibs')
nifticlib_dst = join(nifti_dst, 'nifticlibs')
if not exists(nifticlib_dst):
    mkdir(nifticlib_dst)

def copynifticlib(filename):
    copy(join(nifticlib_src, filename), join(nifticlib_dst, filename))

for nf in nifticlib_list:
    copynifticlib(nf)
