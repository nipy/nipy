"""
In this example, we create a regression model for
an event-related design in which the response
to an event at time T[i] is modeled as depending on 
the amount of time since the last stimulus
T[i-1]
"""

import numpy as np
import nipy.testing as niptest
import sympy

from nipy.modalities.fmri import utils, formula, hrf

dt = np.random.uniform(low=0, high=2.5, size=(50,))
t = np.cumsum(dt)

def alias(expr, name):
    """
    Take an expression of 't' (possibly complicated)
    and make it a '%s(t)' % name, such that
    when it evaluates it has the right values.

    Parameters:
    -----------

    expr : sympy expression, with only 't' as a Symbol

    name : str

    Outputs:
    --------

    nexpr: sympy expression
    
    """
    v = formula.vectorize(expr)
    return formula.aliased_function(name, v)(formula.Term('t'))

a = sympy.Symbol('a')
linear = alias(utils.events(t, dt, f=hrf.glover), 'linear')
quadratic = alias(utils.events(t, dt, f=hrf.glover, g=a**2), 'quad')
cubic = alias(utils.events(t, dt, f=hrf.glover, g=a**3), 'cubic')

f1 = formula.Formula([linear, quadratic, cubic])

# Evaluate them

tval = formula.make_recarray(np.linspace(0,100, 1001), 't')
X1 = f1.design(tval, return_float=True)

# Let's make it exponential with a time constant tau

l = sympy.Symbol('l')
exponential = utils.events(t, dt, f=hrf.glover, g=sympy.exp(-l*a))
f3 = formula.Formula([exponential])

params = formula.make_recarray([(4.5,3.5)], ('l', '_b0'))
X3 = f3.design(tval, params, return_float=True)

# the columns or d/d_b0 and d/dl

tt = tval.view(np.float)
v1 = np.sum([hrf.glovert(tt - s)*np.exp(-4.5*a) for s,a  in zip(t, dt)], 0)
v2 = np.sum([-3.5*a*hrf.glovert(tt - s)*np.exp(-4.5*a) for s,a  in zip(t, dt)], 0)

V = np.array([v1,v2]).T
W = V - np.dot(X3, np.dot(np.linalg.pinv(X3), V))
niptest.assert_almost_equal((W**2).sum() / (V**2).sum(), 0)

