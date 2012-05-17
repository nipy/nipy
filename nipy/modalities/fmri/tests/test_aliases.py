# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module tests nipy's uses of aliased sympy expressions.

That is, sympy.Function's whose value is an arbitrary callable.

We now call these 'implemented functions'.

They've been part of sympy as of 0.7.0

In these tests, the callable's are scipy.interpolate.interp1d instances
representing approximations to Brownian Motions.
"""
import numpy as np

import scipy.interpolate

import sympy

from nipy.fixes.sympy.utilities.lambdify import (implemented_function,
                                                 lambdify)

from nose.tools import assert_true, assert_false, assert_raises

from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_array_almost_equal)


x, y = sympy.symbols(('x', 'y'))


def test_implemented_function():
    # Here we check if the default returned functions are anonymous - in
    # the sense that we can have more than one function with the same name
    f = implemented_function('f', lambda x: 2*x)
    g = implemented_function('f', lambda x: np.sqrt(x))
    l1 = lambdify(x, f(x))
    l2 = lambdify(x, g(x))
    assert_equal(str(f(x)), str(g(x)))
    assert_equal(l1(3), 6)
    assert_equal(l2(3), np.sqrt(3))
    # check that we can pass in a sympy function as input
    func = sympy.Function('myfunc')
    assert_false(hasattr(func, '_imp_'))
    f = implemented_function(func, lambda x: 2*x)
    assert_true(hasattr(func, '_imp_'))


def test_lambdify():
    # Test lambdify with implemented functions
    # first test basic (sympy) lambdify
    f = sympy.cos
    assert_equal(lambdify(x, f(x))(0), 1)
    assert_equal(lambdify(x, 1 + f(x))(0), 2)
    assert_equal(lambdify((x, y), y + f(x))(0, 1), 2)
    # make an implemented function and test
    f = implemented_function("f", lambda x : x+100)
    assert_equal(lambdify(x, f(x))(0), 100)
    assert_equal(lambdify(x, 1 + f(x))(0), 101)
    assert_equal(lambdify((x, y), y + f(x))(0, 1), 101)
    # Error for functions with same name and different implementation
    f2 = implemented_function("f", lambda x : x+101)
    assert_raises(ValueError, lambdify, x, f(f2(x)))
    # our lambdify, like sympy's lambdify, can also handle tuples,
    # lists, dicts as expressions
    lam = lambdify(x, (f(x), x))
    assert_equal(lam(3), (103, 3))
    lam = lambdify(x, [f(x), x])
    assert_equal(lam(3), [103, 3])
    lam = lambdify(x, [f(x), (f(x), x)])
    assert_equal(lam(3), [103, (103, 3)])
    lam = lambdify(x, {f(x): x})
    assert_equal(lam(3), {103: 3})
    lam = lambdify(x, {f(x): x})
    assert_equal(lam(3), {103: 3})
    lam = lambdify(x, {x: f(x)})
    assert_equal(lam(3), {3: 103})


def gen_BrownianMotion():
    X = np.arange(0,5,0.01)
    y = np.random.standard_normal((500,))
    Y = np.cumsum(y)*np.sqrt(0.01)
    B = scipy.interpolate.interp1d(X, Y, bounds_error=0)
    return B


def test_1d():
    B = gen_BrownianMotion()
    Bs = implemented_function("B", B)
    t = sympy.Symbol('t')
    expr = 3*sympy.exp(Bs(t)) + 4
    expected = 3*np.exp(B.y)+4
    ee_vec = lambdify(t, expr, "numpy")
    assert_almost_equal(ee_vec(B.x), expected)
    # with any arbitrary symbol
    b = sympy.Symbol('b')
    expr = 3*sympy.exp(Bs(b)) + 4
    ee_vec = lambdify(b, expr, "numpy")
    assert_almost_equal(ee_vec(B.x), expected)


def test_2d():
    B1, B2 = [gen_BrownianMotion() for _ in range(2)]
    B1s = implemented_function("B1", B1)
    B2s = implemented_function("B2", B2)
    s, t = sympy.symbols(('s', 't'))
    e = B1s(s)+B2s(t)
    ee = lambdify((s,t), e)
    assert_almost_equal(ee(B1.x, B2.x), B1.y + B2.y)
