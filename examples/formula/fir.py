# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import sympy
from nipy.modalities.fmri import formula, utils, hrf

import pylab

t = formula.Term('t')

def linBspline(t, knots):
    """ Create a linear B spline that is zero outside [knots[0],
    knots[-1]] (knots is assumed to be sorted).
    """
    fns = []; symbols=[]
    knots = np.array(knots)
    for i in range(knots.shape[0]-2):
        n = 'bs_%s' % i
        s = sympy.Function(n)
        k1, k2, k3 = knots[i:i+3]
        d1 = k2-k1
        d2 = k3-k2
        def anon(x,k1=k1,k2=k2,k3=k3):
            return ((x-k1) / d1 * np.greater(x, k1) * np.less_equal(x, k2) + 
                    (k3-x) / d1 * np.greater(x, k2) * np.less(x, k3))
        fns.append((n, anon))
        symbols.append(s(t))

    ff = formula.Formula(symbols)
    for n, l in fns:
        ff.aliases[n] = l
    return ff

t = formula.Term('t')
bsp = linBspline(t, np.arange(0,10,2))
tt = np.linspace(0,50,101)
tval = tt.view(np.dtype([('t', np.float)]))

e = np.random.uniform(low=0, high=3, size=(20,)) + 20.
e = np.cumsum(e)
f = formula.Formula([utils.events(e, f=hrf.symbolic(term)) for term in bsp.design])
for k, v in bsp.aliases.items():
    f.aliases[k] = v

d = formula.Design(f, return_float=True)
X = d(tval)

pylab.plot(X[:,0])
pylab.plot(X[:,1])
pylab.plot(X[:,2])
pylab.show()
