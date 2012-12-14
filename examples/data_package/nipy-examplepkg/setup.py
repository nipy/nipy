#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Installation script for nipy examplepkg package '''
import os
from os.path import join as pjoin

from distutils.core import setup

try:
    import ConfigParser as cfg # Python 2
except ImportError:
    import configparser as cfg # Python 3

# The directory under --prefix, under which to store files
OUTPUT_BASE = pjoin('share', 'nipy', 'nipy')

# The directory in this directory to be copied into OUTPUT_BASE
# such that <prefix>/<OUPUT_BASE>/<PKG_BASE> will exist
PKG_BASE = 'examplepkg'

DATA_FILES = []

for dirpath, dirnames, filenames in os.walk(PKG_BASE):
    files = [pjoin(dirpath, filename) for filename in filenames]
    DATA_FILES.append((pjoin(OUTPUT_BASE, dirpath), files))

config = cfg.SafeConfigParser()
config.read(pjoin(PKG_BASE, 'config.ini'))

setup(
    name = 'nipy-' + PKG_BASE,
    version = config.get('DEFAULT', 'version'),
    description='NIPY %s data package' % PKG_BASE,
    author='The NIPY team',
    url='http://neuroimaging.scipy.org',
    author_email='nipy-devel@neuroimaging.scipy.org',
    data_files = DATA_FILES,
    )

