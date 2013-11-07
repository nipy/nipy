# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Package containing generic algorithms such as registration, statistics,
simulation, etc.
"""
__docformat__ = 'restructuredtext'

import statistics
import fwhm, interpolation, kernel_smooth, diagnostics

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench

# AR, Sep 29 2013: I am copying that from former __init__ file from
# nipy/labs without knowing whether this is useful.
# Import here only files that don't draw in compiled code: that way the
# basic functionality is still usable even if the compiled
# code is messed up (32/64 bit issues, or binary incompatibilities)
from .mask import compute_mask_files, compute_mask_sessions, \
            series_from_mask
