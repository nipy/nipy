# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Generic algorithms such as registration, statistics, simulation, etc.
"""
__docformat__ = 'restructuredtext'

from nipy.testing import Tester

from . import diagnostics, fwhm, interpolation, kernel_smooth, statistics

test = Tester().test
bench = Tester().bench
