# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
An (approximate) version of SPM's run-level model for fMRI data

Consists of an OLS pass through the data, followed by a pooled estimate
of a covariance matrix constructed from a series expansion of an
AR1 model, expanded in terms of rho.
"""
from __future__ import absolute_import

from . import model
