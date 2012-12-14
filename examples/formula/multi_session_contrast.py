#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Example of more than one run in the same model
"""
from __future__ import print_function # Python 2/3 compatibility

import numpy as np

from nipy.algorithms.statistics.api import Term, Formula, Factor
from nipy.modalities.fmri import utils, hrf

# HRF models we will use for each run.  Just to show it can be done, use a
# different HRF model for each run
h1 = hrf.glover
h2 = hrf.afni

# Symbol for time in general.  The 'events' function below will return models in
# terms of 't', but we'll want models in terms of 't1' and 't2'.  We need 't'
# here so we can substitute.
t = Term('t')

# run 1
t1 = Term('t1') # Time within run 1
c11 = utils.events([3, 7, 10], f=h1) # Condition 1, run 1
# The events utility returns a formula in terms of 't' - general time
c11 = c11.subs(t, t1) # Now make it in terms of time in run 1
# Same for conditions 2 and 3
c21 = utils.events([1, 3, 9], f=h1); c21 = c21.subs(t, t1)
c31 = utils.events([2, 4, 8], f=h1); c31 = c31.subs(t, t1)
# Add also a Fourier basis set for drift with frequencies 0.3, 0.5, 0.7
d1 = utils.fourier_basis([0.3, 0.5, 0.7]); d1 = d1.subs(t, t1)

# Here's our formula for run 1 signal terms of time in run 1 (t1)
f1 = Formula([c11,c21,c31]) + d1

# run 2
t2 = Term('t2') # Time within run 2
# Conditions 1 through 3 in run 2
c12 = utils.events([3.3, 7, 10], f=h2); c12 = c12.subs(t, t2)
c22 = utils.events([1, 3.2, 9], f=h2); c22 = c22.subs(t, t2)
c32 = utils.events([2, 4.2, 8], f=h2); c32 = c32.subs(t, t2)
d2 = utils.fourier_basis([0.3, 0.5, 0.7]); d2 = d2.subs(t, t2)

# Formula for run 2 signal in terms of time in run 2 (t2)
f2 = Formula([c12, c22, c32]) + d2

# Factor giving constant for run. The [1, 2] means that there are two levels to
# this factor, and that when we get to pass in values for this factor,
# instantiating an actual design matrix (see below), a value of 1 means level
# 1 and a value of 2 means level 2.
run_factor = Factor('run', [1, 2])
run_1_coder = run_factor.get_term(1) # Term coding for level 1
run_2_coder = run_factor.get_term(2) # Term coding for level 2

# The multi run formula will combine the indicator (dummy value) terms from the
# run factor with the formulae for the runs (which are functions of (run1, run2)
# time. The run_factor terms are step functions that are zero when not in the
# run, 1 when in the run.
f = Formula([run_1_coder]) * f1 + Formula([run_2_coder]) * f2 + run_factor

# Now, we evaluate the formula.  So far we've been entirely symbolic.  Now we
# start to think about the values at which we want to evaluate our symbolic
# formula.

# We'll use these values for time within run 1.  The times are in seconds from
# the beginning of run 1.  In our case run 1 was 20 seconds long. 101 below
# gives 101 values from 0 to 20 including the endpoints, giving a dt of 0.2.
tval1 = np.linspace(0, 20, 101)
# run 2 lasts 10 seconds.  These are the times in terms of the start of run 2.
tval2 = np.linspace(0, 10, 51)

# We pad out the tval1 / tval2 time vectors with zeros corresponding to the
# TRs in run 2 / run 1.
ttval1 = np.hstack([tval1, np.zeros(tval2.shape)])
ttval2 = np.hstack([np.zeros(tval1.shape), tval2])
# The arrays above now have 152=101+51 rows...

# Vector of run numbers for each time point (with values 1 or 2)
run_no = np.array([1]*tval1.shape[0] + [2]*tval2.shape[0])

# Create the recarray that will be used to create the design matrix. The
# recarray gives the actual values for the symbolic terms in the formulae.  In
# our case the terms are t1, t2, and the (indicator coding) terms from the run
# factor.
rec = np.array([(tv1, tv2, s) for tv1, tv2, s in zip(ttval1, ttval2, run_no)],
               np.dtype([('t1', np.float),
                         ('t2', np.float),
                         ('run', np.int)]))

# The contrast we care about
contrast = Formula([run_1_coder * c11 - run_2_coder * c12])

# # Create the design matrix
X = f.design(rec, return_float=True)

# Show ourselves the design space covered by the contrast, and the corresponding
# contrast matrix
preC = contrast.design(rec, return_float=True)
# C is the matrix such that preC = X.dot(C.T)
C = np.dot(np.linalg.pinv(X), preC)
print(C)

# We can also get this by passing the contrast into the design creation.
X, c = f.design(rec, return_float=True, contrasts=dict(C=contrast))
assert np.allclose(C, c['C'])

# Show the names of the non-trivial elements of the contrast
nonzero = np.nonzero(np.fabs(C) >= 1e-5)[0]
print((f.dtype.names[nonzero[0]], f.dtype.names[nonzero[1]]))
print(((run_1_coder * c11), (run_2_coder * c12)))
