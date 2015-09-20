# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module is meant to reproduce the GLM analysis of fmristat.

    Liao et al. (2002).
TODO fix reference here

"""
from __future__ import absolute_import
__docformat__ = 'restructuredtext'

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
