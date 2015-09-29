# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Generic algorithms such as registration, statistics, simulation, etc.
"""
from __future__ import absolute_import
__docformat__ = 'restructuredtext'

from . import statistics
from . import fwhm, interpolation, kernel_smooth, diagnostics

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
