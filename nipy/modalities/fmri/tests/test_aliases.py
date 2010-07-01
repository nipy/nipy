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

from nipy.modalities.fmri.aliased import (aliased_function,
                                          vectorize,
                                          lambdify)

from nose.tools import assert_true, assert_false, assert_raises

from numpy.testing import assert_almost_equal, assert_equal, \
    assert_array_almost_equal

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
    Bs = aliased_function("B", B)
    t = sympy.Symbol('t')
    def_t = sympy.DeferredVector('t')
    def_expr = 3*sympy.exp(Bs(def_t)) + 4
    expected = 3*np.exp(B.y)+4
    ee_lam = lambdify(def_t, def_expr)
    yield assert_almost_equal(ee_lam(B.x), expected)
    # use vectorize to do the deferred stuff
    expr = 3*sympy.exp(Bs(t)) + 4
    ee_vec = vectorize(expr)
    yield assert_almost_equal(ee_vec(B.x), expected)
    # with any arbitrary symbol
    b = sympy.Symbol('b')
    expr = 3*sympy.exp(Bs(b)) + 4
    ee_vec = vectorize(expr, b)
    yield assert_almost_equal(ee_vec(B.x), expected)
    

@parametric
def test_2d():
    B1, B2 = [gen_BrownianMotion() for _ in range(2)]
    B1s = aliased_function("B1", B1)
    B2s = aliased_function("B2", B2)
    t = sympy.DeferredVector('t')
    s = sympy.DeferredVector('s')
    e = B1s(s)+B2s(t)
    ee = lambdify((s,t), e)
    yield assert_almost_equal(ee(B1.x, B2.x), B1.y + B2.y)


@parametric
def test_alias_anon():
    # Here we check if the default returned functions are anonymous - in
    # the sense that we can have more than one function with the same name
    f = aliased_function('f', lambda x: 2*x)
    g = aliased_function('f', lambda x: np.sqrt(x))
    x = sympy.Symbol('x')
    l1 = lambdify(x, f(x))
    l2 = lambdify(x, g(x))
    yield assert_equal(str(f(x)), str(g(x)))
    yield assert_equal(l1(3), 6)
    yield assert_equal(l2(3), np.sqrt(3))


@parametric
def test_func_input():
    # check that we can pass in a sympy function as input
    func = sympy.Function('myfunc')
    yield assert_false(hasattr(func, 'alias'))
    f = aliased_function(func, lambda x: 2*x)
    yield assert_true(hasattr(func, 'alias'))


@parametric
def test_vectorize():
    theta = sympy.Symbol('theta')
    num_func = lambdify(theta, sympy.cos(theta))
    yield assert_equal(num_func(0), 1)
    # we don't need to do anything for a naturally numpy'ed function
    yield assert_array_almost_equal(num_func([0, np.pi]), [1, -1])
    # but we do for single valued functions
    func = aliased_function('f', lambda x: x**2)
    num_func = lambdify(theta, func(theta))
    yield assert_equal(num_func(2), 4)
    # ** on a list raises a type error
    yield assert_raises(TypeError, num_func, [2, 3])
    # so vectorize
    num_func = vectorize(func(theta), theta)


@parametric
def test_alias_tuple():
    # lambdify can also handle tuples, lists, dicts as expressions
    f = aliased_function("f", lambda x : x+100)
    t = sympy.Symbol('t')
    lam = lambdify(t, (f(t), t))
    yield assert_equal(lam(3), (103, 3))
    lam = lambdify(t, [f(t), t])
    yield assert_equal(lam(3), [103, 3])
    lam = lambdify(t, [f(t), (f(t), t)])
    yield assert_equal(lam(3), [103, (103, 3)])
    lam = lambdify(t, {f(t): t})
    yield assert_equal(lam(3), {103: 3})
    lam = lambdify(t, {f(t): t})
    yield assert_equal(lam(3), {103: 3})
    lam = lambdify(t, {t: f(t)})
    yield assert_equal(lam(3), {3: 103})
