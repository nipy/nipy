# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" This example is now out of date - please remind us to fix it
"""
import numpy as np
import sympy

from nipy.algorithms.statistics.api import Term, Formula, Factor
from nipy.modalities.fmri import utils, hrf

# hrf

h1 = sympy.Function('hrf1')
h2 = sympy.Function('hrf2')
t = Term('t')

# Session 1

t1 = Term('t1')
c11 = utils.events([3,7,10], f=h1); c11 = c11.subs(t, t1)
c21 = utils.events([1,3,9], f=h1); c21 = c21.subs(t, t1)
c31 = utils.events([2,4,8], f=h1); c31 = c31.subs(t, t1)
d1 = utils.fourier_basis([0.3,0.5,0.7]); d1 = d1.subs(t, t1)
tval1 = np.linspace(0,20,101)

f1 = Formula([c11,c21,c31]) + d1
f1 = f1.subs('hrf1', hrf.glover)

# Session 2

t2 = Term('t2')
c12 = utils.events([3.3,7,10], f=h2); c12 = c12.subs(t, t2)
c22 = utils.events([1,3.2,9], f=h2); c22 = c22.subs(t, t2)
c32 = utils.events([2,4.2,8], f=h2); c32 = c32.subs(t, t2)
d2 = utils.fourier_basis([0.3,0.5,0.7]); d2 = d2.subs(t, t2)
tval2 = np.linspace(0,10,51)

f2 = Formula([c12,c22,c32]) + d2
f2 = f2.subs('hrf2', hrf.dglover)

sess_factor = Factor('sess', [1,2])

# The multi session formula

f = Formula([sess_factor.terms[0]]) * f1 + Formula([sess_factor.terms[1]]) * f2

# Now, we evaluate the formula
# the arrays now have 152=101+51 rows...

ttval1 = np.hstack([tval1, np.zeros(tval2.shape)])
ttval2 = np.hstack([np.zeros(tval1.shape), tval2])
session = np.array([1]*tval1.shape[0] + [2]*tval2.shape[0])

f.subs('hrf1', hrf.glover)
f.subs('hrf2', hrf.dglover)

# Create the recarray that will be used to create the design matrix

rec = np.array([(t1,t2, s) for t1, t2, s in zip(ttval1, ttval2, session)],
               np.dtype([('t1', np.float),
                         ('t2', np.float),
                         ('sess', np.int)]))

# The contrast we care about

# It would be a good idea to be able to create
# the contrast from the Formula, "f" above,
# applying all of f's aliases to it....

# contrast = Formula([c11-c12])
# contrast.aliases['hrf1'] = hrf.glover
# contrast.aliases['hrf2'] = hrf.dglover

# # Create the design matrix

# X = d(rec, return_float=True)

# d2 = Design(contrast, return_float=True)
# preC = d2(rec)

# C = np.dot(np.linalg.pinv(X), preC)
# print C
