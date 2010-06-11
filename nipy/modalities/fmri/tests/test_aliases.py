# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module tests nipy's uses of aliased sympy expressions.

That is, sympy.Function's whose value is an arbitrary callable.

In these tests, the callable's are scipy.interpolate.interp1d instances
representing approximations to Brownian Motions.

"""
import numpy as np

import scipy.interpolate

import sympy

from nipy.modalities.fmri import formula, aliased
from nipy.modalities.fmri.aliased import aliased_function

from numpy.testing import assert_almost_equal, assert_equal

from nipy.testing import parametric


def gen_BrownianMotion():
    X = np.arange(0,5,0.01)
    y = np.random.standard_normal((500,))
    Y = np.cumsum(y)*np.sqrt(0.01)
    B = scipy.interpolate.interp1d(X, Y, bounds_error=0)
    return B


@parametric
def test_1d():
    B = gen_BrownianMotion()
    Bs = formula.aliased_function("B", B)
    t = sympy.DeferredVector('t')
    n={};
    aliased._add_aliases_to_namespace(n, Bs)
    expr = 3*sympy.exp(Bs(t)) + 4
    ee = sympy.lambdify(t, expr, (n, 'numpy'))
    yield assert_almost_equal(ee(B.x), 3*np.exp(B.y)+4)


@parametric
def test_2d():
    B1, B2 = [gen_BrownianMotion() for _ in range(2)]
    B1s = formula.aliased_function("B1", B1)
    B2s = formula.aliased_function("B2", B2)
    t = sympy.DeferredVector('t')
    s = sympy.DeferredVector('s')
    e = B1s(s)+B2s(t)
    n={};
    aliased._add_aliases_to_namespace(n, e)
    ee = sympy.lambdify((s,t), e, (n, 'numpy'))
    yield assert_almost_equal(ee(B1.x, B2.x), B1.y + B2.y)


@parametric
def test_alias2():
    f = aliased_function('f', lambda x: 2*x)
    g = aliased_function('f', lambda x: np.sqrt(x))
    x = sympy.Symbol('x')
    l1 = aliased.lambdify(x, f(x))
    l2 = aliased.lambdify(x, g(x))
    yield assert_equal(str(f(x)), str(g(x)))
    yield assert_equal(l1(3), 6)
    yield assert_equal(l2(3), np.sqrt(3))


