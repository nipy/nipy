#!/usr/bin/env python
"""Copy source files from pynifti git directory into nipy source directory.
We only want to copy the files necessary to build pynifti and the nifticlibs,
and use them within nipy.  We will not copy docs, tests, etc...

Pynifti should be build before this script is run so swig generates the 
wrapper for nifticlib.  We do not want swig as a dependency for nipy.

"""

from os import mkdir
from os.path import join, exists
from shutil import copy2 as copy

"""
The pynifti source should be in a directory level with nipy-trunk
Ex: 
    /Users/cburns/src/nipy
    /Users/cburns/src/pynifti

"""

src_dir = '../../../../../pynifti'
dst_dir = 'temp'

assert exists(src_dir)

copy(join(src_dir, 'AUTHOR'), join(dst_dir, 'AUTHOR'))
copy(join(src_dir, 'COPYING'), join(dst_dir, 'COPYING'))
nifti_src = join(src_dir, 'nifti')
nifti_dst = join(dst_dir, 'nifti')
def copynifti(filename):
    copy(join(nifti_src, filename), join(nifti_dst, filename))

nifti_list = ['niftiformat.py', 'niftiimage.py', 'utils.py',
              'nifticlib.py', 'nifticlib_wrap.c']
if not exists(nifti_dst):
    mkdir(nifti_dst)
for nf in nifti_list:
    copynifti(nf)

nifticlib_src = join(src_dir, '3rd', 'nifticlibs')
nifticlib_dst = join(nifti_dst, 'nifticlibs')
def copynifticlib(filename):
    copy(join(nifticlib_src, filename), join(nifticlib_dst, filename))

nifticlib_list = ['LICENSE', 'README', 'nifti1.h', 'nifti1_io.c',
                  'nifti1_io.h', 'znzlib.c', 'znzlib.h'] 
if not exists(nifticlib_dst):
    mkdir(nifticlib_dst)
for nf in nifticlib_list:
    copynifticlib(nf)

"""
Files we currently use:

AUTHOR
COPYING
nifti/
    __init__.py
    nifticlib.py    (swig generated)
    nifticlib_wrap.c    (swig generated)
    niftiimage.py
    nifticlib/
        include/*
        niftilib/*
        znzlib/*

# From new pynifti source... June 2008
pynifti/
    AUTHOR
    COPYING
    ** setup.py   (write our own)
    nifti/
        niftiformat.py
        niftiimage.py
        utils.py
        nifticlib.py    (swig)
        nifticlib_wrap.c    (swig)
        ** __init__.py    (write our own to control 'import nifti' in *.py's)

    3rd/nifticlibs
        LICENSE
        README
        nifti1.h
        nifti1_io.c, h
        znzlib.c, h

cburns@pynifti 11:47:49 $ pwd
/Users/cburns/src/pynifti
cburns@pynifti 11:47:54 $ ls -l nifti
total 152
-rw-r--r--  1 cburns  staff   1162 Jun 26 14:32 __init__.py
-rw-r--r--  1 cburns  staff   6210 Jun 26 14:32 nifticlib.i
-rw-r--r--  1 cburns  staff  28301 Jun 26 14:32 niftiformat.py
-rw-r--r--  1 cburns  staff  17686 Jun 26 14:32 niftiimage.py
-rw-r--r--  1 cburns  staff  15867 Jun 26 14:32 utils.py
cburns@pynifti 11:47:57 $ ls -l 3rd/nifticlibs/
total 736
-rw-r--r--  1 cburns  staff     388 Jun 26 14:32 LICENSE
-rw-r--r--  1 cburns  staff     302 Jun 26 14:32 Makefile
-rwxr-xr-x  1 cburns  staff     302 Jun 26 14:32 Makefile.win
-rw-r--r--  1 cburns  staff     653 Jun 26 14:32 README
-rw-r--r--  1 cburns  staff   69236 Jun 26 14:32 nifti1.h
-rw-r--r--  1 cburns  staff  254161 Jun 26 14:32 nifti1_io.c
-rw-r--r--  1 cburns  staff   20250 Jun 26 14:32 nifti1_io.h
-rw-r--r--  1 cburns  staff    6652 Jun 26 14:32 znzlib.c
-rw-r--r--  1 cburns  staff    3028 Jun 26 14:32 znzlib.h
"""


"""
Don't copy test directory, run tests in pynifti git repos!

# copy test directory
test_dir = os.path.join(src_dir, 'tests')
assert os.path.exists(test_dir)
print '\ncopy tests directory:', os.path.abspath(test_dir)
for dirpath, dirnames, filenames in os.walk(test_dir):
    dst_dir = os.path.join(dst_dir, os.path.basename(dirpath))
    if not os.path.exists(dst_dir):
        # Make destination directory if it doesn't exist
        os.mkdir(dst_dir)
    for file_ in filenames:
        src = os.path.join(dirpath, file_)
        print 'copy src:', src
        dst = os.path.join(dst_dir, file_)
        print 'to dst:', dst
        shutil.copy2(src, dst)
"""
