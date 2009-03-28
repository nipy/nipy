"""The testing directory contains a small set of imaging files to be used
for doctests only.  More thorough tests and example data will be stored in
a nipy-data-suite to be created later and downloaded separately.

Examples
--------

>>> from neuroimaging.testing import funcfile
>>> from neuroimaging.core.image import image
>>> img = image.load(funcfile)
>>> img.shape
(20, 2, 20, 20)

Notes
-----
BUG: anatomical.nii.gz is a copy of functional.nii.gz.  This is a place-holder
    until we build a proper anatomical test image.

"""

import os

#__all__ = ['funcfile', 'anatfile']

# Discover directory path
filepath = os.path.abspath(__file__)
basedir = os.path.dirname(filepath)

funcfile = os.path.join(basedir, 'functional.nii.gz')
anatfile = os.path.join(basedir, 'anatomical.nii.gz')

from numpy.testing import *
import decorators as dec
from nose.tools import assert_true, assert_false

