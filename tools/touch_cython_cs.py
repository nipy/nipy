#!/usr/bin/env python
""" Refresh modification times on Cython generated C files

This is sometimes necessary on windows when the git checkout appears to
sometimes checkout the C files with modification times earlier than the pyx
files, triggering an attempt to rebuild the C files with Cython when running a
build.
"""
from __future__ import with_statement
import os
from os.path import splitext, join as pjoin, isfile
import sys
import optparse


# From http://stackoverflow.com/questions/1158076/implement-touch-using-python
if sys.version_info[0] >= 3:
    def touch(fname, times=None, ns=None, dir_fd=None):
        with os.open(fname, os.O_APPEND, dir_fd=dir_fd) as f:
            os.utime(f.fileno() if os.utime in os.supports_fd else fname,
                times=times, ns=ns, dir_fd=dir_fd)
else:
    def touch(fname, times=None):
        with file(fname, 'a'):
            os.utime(fname, times)


def main():
    parser = optparse.OptionParser(usage='%prog [<root_dir>]')
    (opts, args) = parser.parse_args()
    if len(args) > 1:
        parser.print_help()
        sys.exit(-1)
    elif len(args) == 1:
        root_dir = args[0]
    else:
        root_dir = os.getcwd()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith('.pyx'):
                froot, ext = splitext(fn)
                cfile = pjoin(dirpath, froot + '.c')
                if isfile(cfile):
                    touch(cfile)


if __name__ == '__main__':
    main()
