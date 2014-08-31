#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Example of FIR model using formula framework

Shows how to use B splines as basis functions for the FIR instead of simple
boxcars.

Requires matplotlib
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from sympy.utilities.lambdify import implemented_function

from nipy.algorithms.statistics.api import Formula
from nipy.modalities.fmri import utils

def linBspline(knots):
    """ Create linear B spline that is zero outside [knots[0], knots[-1]]

    (knots is assumed to be sorted).
    """
    fns = []
    knots = np.array(knots)
    for i in range(knots.shape[0]-2):
        name = 'bs_%s' % i
        k1, k2, k3 = knots[i:i+3]
        d1 = k2-k1
        def anon(x,k1=k1,k2=k2,k3=k3):
            return ((x-k1) / d1 * np.greater(x, k1) * np.less_equal(x, k2) +
                    (k3-x) / d1 * np.greater(x, k2) * np.less(x, k3))
        fns.append(implemented_function(name, anon))
    return fns


# The splines are functions of t (time)
bsp_fns = linBspline(np.arange(0,10,2))

# We're going to evaluate at these specific values of time
tt = np.linspace(0,50,101)
tvals= tt.view(np.dtype([('t', np.float)]))

# Some inter-stimulus intervals
isis = np.random.uniform(low=0, high=3, size=(4,)) + 10.

# Made into event onset times
e = np.cumsum(isis)

# Make event onsets into functions of time convolved with the spline functions.
event_funcs = [utils.events(e, f=fn) for fn in bsp_fns]

# Put into a formula.
f = Formula(event_funcs)

# The design matrix
X = f.design(tvals, return_float=True)

# Show the design matrix as line plots
plt.plot(X[:,0])
plt.plot(X[:,1])
plt.plot(X[:,2])
plt.xlabel('time (s)')
plt.title('B spline used as bases for an FIR response model')
plt.show()
