"""
This module tests nipy's uses of aliased sympy expressions.

That is, sympy.Function's whose value is an arbitrary callable.

In these tests, the callable's are scipy.interpolate.interp1d instances
representing approximations to Brownian Motions.

"""
import numpy as np
import scipy.interpolate
import pylab
import sympy
from neuroimaging.modalities.fmri import utils

def gen_BrownianMotion():
    X = np.arange(0,5,0.01)
    y = np.random.standard_normal((500,))
    Y = np.cumsum(y)*np.sqrt(0.01)
    B = scipy.interpolate.interp1d(X, Y, bounds_error=0)
    return B

def test_1d():

    B = gen_BrownianMotion()
    Bs = sympy.Function('B')
    t = sympy.DeferredVector('t')

    utils.set_alias(Bs, B)
    n={}; utils.add_aliases_to_namespace(Bs, n)
    utils.add_aliases_to_namespace(Bs(t), n)

    expr = 3*sympy.exp(Bs(t)) + 4
    ee = sympy.lambdify(t, expr, (n, 'numpy'))

    np.testing.assert_almost_equal(ee(B.x), 3*np.exp(B.y)+4)

def test_2d():

    B1, B2 = [gen_BrownianMotion() for _ in range(2)]
    B1s = sympy.Function('B1')
    B2s = sympy.Function('B2')
    t = sympy.DeferredVector('t')
    s = sympy.DeferredVector('s')

    utils.set_alias(B1s, B1)
    utils.set_alias(B2s, B2)

    e = B1s(s)+B2s(t)
    n={}; utils.add_aliases_to_namespace(e, n)

    ee = sympy.lambdify((s,t), e, (n, 'numpy'))
    np.testing.assert_almost_equal(ee(B1.x, B2.x), B1.y + B2.y)
