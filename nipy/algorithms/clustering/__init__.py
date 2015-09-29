# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This sub-package contains functions for clustering.
It might be removed in the future, and replaced 
by an optional dependence on scikit learn. 
"""
from __future__ import absolute_import

from nipy.testing import Tester

test = Tester().test
bench = Tester().bench 
