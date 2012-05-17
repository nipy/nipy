# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing fmri utils

"""

import numpy as np

import sympy
from sympy import Symbol, Function, DiracDelta
from nipy.fixes.sympy.utilities.lambdify import lambdify

from nipy.algorithms.statistics.formula import Term

from ..utils import (
    lambdify_t,
    define,
    events,
    blocks,
    interp,
    linear_interp,
    step_function,
    convolve_functions,
    )
from .. import hrf

from nose.tools import assert_equal, raises

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)


t = Term('t')


def test_define():
    expr = sympy.exp(3*t)
    assert_equal(str(expr), 'exp(3*t)')
    newf = define('f', expr)
    assert_equal(str(newf), 'f(t)')
    f = lambdify_t(newf)
    tval = np.random.standard_normal((3,))
    assert_almost_equal(np.exp(3*tval), f(tval))


def test_events():
    # test events utility function
    h = Function('hrf')
    evs = events([3,6,9])
    assert_equal(DiracDelta(-9 + t) + DiracDelta(-6 + t) +
                 DiracDelta(-3 + t), evs)
    evs = events([3,6,9], f=h)
    assert_equal(h(-3 + t) + h(-6 + t) + h(-9 + t), evs)
    # make some beta symbols
    b = [Symbol('b%d' % i, dummy=True) for i in range(3)]
    a = Symbol('a')
    p = b[0] + b[1]*a + b[2]*a**2
    evs = events([3,6,9], amplitudes=[2,1,-1], g=p)
    assert_equal((2*b[1] + 4*b[2] + b[0])*DiracDelta(-3 + t) +
                 (-b[1] + b[0] + b[2])*DiracDelta(-9 + t) +
                 (b[0] + b[1] + b[2])*DiracDelta(-6 + t),
                 evs)
    evs = events([3,6,9], amplitudes=[2,1,-1], g=p, f=h)
    assert_equal((2*b[1] + 4*b[2] + b[0])*h(-3 + t) +
                 (-b[1] + b[0] + b[2])*h(-9 + t) +
                 (b[0] + b[1] + b[2])*h(-6 + t),
                 evs)
    # test no error for numpy int arrays
    onsets = np.array([30, 70, 100], dtype=np.int64)
    evs = events(onsets, f=hrf.glover)


def test_interp():
    times = [0,4,5.]
    values = [2.,4,6]
    for int_func in (interp, linear_interp):
        s = int_func(times, values, bounds_error=False)
        tval = np.array([-0.1,0.1,3.9,4.1,5.1])
        res = lambdify(t, s)(tval)
        assert_array_equal(np.isnan(res),
                           [True, False, False, False, True])
        assert_array_almost_equal(res[1:-1], [2.05, 3.95, 4.2])
        # specifying kind as linear is OK
        s = linear_interp(times, values, kind='linear')


@raises(ValueError)
def test_linear_inter_kind():
    linear_interp([0, 1], [1, 2], kind='cubic')


def test_step_function():
    # test step function
    # step function is a function of t
    s = step_function([0,4,5],[2,4,6])
    tval = np.array([-0.1,0,3.9,4,4.1,5.1])
    lam = lambdify(t, s)
    assert_array_equal(lam(tval), [0, 2, 2, 4, 4, 6])
    s = step_function([0,4,5],[4,2,1])
    lam = lambdify(t, s)
    assert_array_equal(lam(tval), [0, 4, 4, 2, 2, 1])


def test_blocks():
    on_off = [[1,2],[3,4]]
    tval = np.array([0.4,1.4,2.4,3.4])
    b = blocks(on_off)
    lam = lambdify(t, b)
    assert_array_equal(lam(tval), [0, 1, 0, 1])
    b = blocks(on_off, amplitudes=[3,5])
    lam = lambdify(t, b)
    assert_array_equal(lam(tval), [0, 3, 0, 5])


def numerical_convolve(func1, func2, interval, dt, padding_f=0.1):
    mni, mxi = interval
    diff = mxi - mni
    pad_t = diff * padding_f
    time = np.arange(mni, mxi+pad_t, dt)
    vec1 = func1(time)
    vec2 = func2(time)
    value = np.convolve(vec1, vec2) * dt
    min_s = min(time.size, value.size)
    time = time[:min_s]
    value = value[:min_s]
    return time, value


def test_convolve_functions():
    # replicate convolution
    # This is a square wave on [0,1]
    f1 = (t > 0) * (t < 1)
    # ff1 is the numerical implementation of same
    lam_f1 = lambdify(t, f1)
    ff1 = lambda x : lam_f1(x).astype(np.int)
    # The convolution of ``f1`` with itself is a triangular wave on
    # [0,2], peaking at 1 with height 1
    tri = convolve_functions(f1, f1, [0,2], 1.0e-3, name='conv')
    assert_equal(str(tri), 'conv(t)')
    ftri = lambdify(t, tri)
    time, value = numerical_convolve(ff1, ff1, [0, 2], 1.0e-3)
    y = ftri(time)
    # numerical convolve about the same as ours
    assert_array_almost_equal(value, y)
    # peak is at 1
    assert_array_almost_equal(time[np.argmax(y)], 1)
    # Flip the interval and get the same result
    tri = convolve_functions(f1, f1, [2, 0], 1.0e-3)
    ftri = lambdify(t, tri)
    y = ftri(time)
    assert_array_almost_equal(value, y)
    # offset square wave by 1
    f2 = (t > 1) * (t < 2)
    tri = convolve_functions(f1, f2, [0,3], 1.0e-3)
    ftri = lambdify(t, tri)
    y = ftri(time)
    assert_array_almost_equal(time[np.argmax(y)], 2)
    # offset both by 1 and start interval at one
    tri = convolve_functions(f2, f2, [1,3], 1.0e-3)
    ftri = lambdify(t, tri)
    # get numerical version
    lam_f2 = lambdify(t, f2)
    ff2 = lambda x : lam_f2(x).astype(np.int)
    time, value = numerical_convolve(ff2, ff2, [1, 3], 1.0e-3)
    # and our version, compare
    y = ftri(time)
    assert_array_almost_equal(y, value)
