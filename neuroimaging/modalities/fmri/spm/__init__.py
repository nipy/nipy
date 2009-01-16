"""
An (approximate) version of SPM's run-level model for fMRI data

Consists of an OLS pass through the data, followed by a pooled estimate
of a covariance matrix constructed from a series expansion of an
AR1 model, expanded in terms of rho.
"""

import model
