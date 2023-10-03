# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The testing directory contains a small set of imaging files to be
used for doctests only.  More thorough tests and example data will be
stored in a nipy data packages that you can download separately.

.. note:

   We use the ``pytest`` testing framework for tests.

   ``pytest`` is a dependency for the tests, but should not be a dependency for
   running the algorithms in the NIPY library.  This file should import without
   ``pytest`` being present on the python path.

Examples
--------

>>> from nipy.testing import funcfile
>>> from nipy.io.api import load_image
>>> img = load_image(funcfile)
>>> img.shape
(17, 21, 3, 20)

"""

import os

# Discover directory path
filepath = os.path.abspath(__file__)
basedir = os.path.dirname(filepath)

funcfile = os.path.join(basedir, 'functional.nii.gz')
anatfile = os.path.join(basedir, 'anatomical.nii.gz')

from numpy.testing import *

from . import decorators as dec
