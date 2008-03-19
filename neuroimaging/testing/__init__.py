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

"""

import os

__all__ = ['funcfile']

# Discover directory path
filepath = os.path.abspath(__file__)
basedir = os.path.dirname(filepath)

funcfile = os.path.join(basedir, 'functional.nii.gz')
