"""
In this example, we create a regression model for
an event-related design in which the response
to an event at time T[i] is modeled as depending on 
the amount of time since the last stimulus
T[i-1]
"""

import numpy as np
import sympy

from neuroimaging.modalities.fmri import utils, formula, hrf

dt = np.random.uniform(low=0, high=2.5, size=(50,))
t = np.cumsum(dt)

a = sympy.Symbol('a')
linear = utils.events(t, dt, f=hrf.glover_sympy)
quadratic = utils.events(t, dt, f=hrf.glover_sympy, g=a**2)
cubic = utils.events(t, dt, f=hrf.glover_sympy, g=a**3)

f1 = formula.Formula([linear, quadratic, cubic])

# Another way

vlinear = utils.Vectorize(linear)
vquadratic = utils.Vectorize(quadratic)
vcubic = utils.Vectorize(cubic)

flinear = sympy.Function('linear')
fquadratic = sympy.Function('quadratic')
fcubic = sympy.Function('cubic')

f2 = formula.Formula([flinear(utils.t), fquadratic(utils.t), fcubic(utils.t)])
print f2.mean

f2.aliases['linear'] = vlinear
f2.aliases['quadratic'] = vquadratic
f2.aliases['cubic'] = vcubic

# Evaluate them

tval = np.linspace(0,100, 1001).view(np.dtype([('t', np.float)]))
d1 = formula.Design(f1, return_float=True)
d2 = formula.Design(f2, return_float=True)

X1 = d1(tval)
X2 = d2(tval)

np.testing.assert_almost_equal(X1, X2)

# Let's make it exponential with a time constant tau

l = sympy.Symbol('l')
exponential = utils.events(t, dt, f=sympy.Function('hrf'), g=sympy.exp(-l*a))
f3 = formula.Formula([exponential])
f3.aliases['hrf'] = hrf.glover # this is just a callable, not symbolic
d3 = formula.Design(f3, return_float=True)

params = np.array([(4.5,3.5)], np.dtype([('l', np.float),
                                         ('_b0', np.float)]))
X3 = d3(tval, params)
