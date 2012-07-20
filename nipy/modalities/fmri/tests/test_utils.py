# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing fmri utils

"""
import re

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

from nose.tools import (assert_equal, assert_false, raises, assert_raises)

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
        s = int_func(times, values, np.nan)
        tval = np.array([-0.1,0.1,3.9,4.1,5.1])
        res = lambdify(t, s)(tval)
        assert_array_equal(np.isnan(res),
                           [True, False, False, False, True])
        assert_array_almost_equal(res[1:-1], [2.05, 3.95, 4.2])
        # default is zero fill
        s = int_func(times, values)
        res = lambdify(t, s)(tval)
        assert_array_almost_equal(res, [0, 2.05, 3.95, 4.2, 0])
        # Can be some other value
        s = int_func(times, values, fill=10)
        res = lambdify(t, s)(tval)
        assert_array_almost_equal(res, [10, 2.05, 3.95, 4.2, 10])
        # If fill is None, raises error on interpolation outside bounds
        s = int_func(times, values, fill=None)
        f = lambdify(t, s)
        assert_array_almost_equal(f(tval[1:-1]), [2.05, 3.95, 4.2])
        assert_raises(ValueError, f, tval[:-1])
        # specifying kind as linear is OK
        s = linear_interp(times, values, kind='linear')
        # bounds_check should match fill
        int_func(times, values, bounds_error=False)
        int_func(times, values, fill=None, bounds_error=True)
        assert_raises(ValueError, int_func, times, values, bounds_error=True)
        # fill should match fill value
        int_func(times, values, fill=10, fill_value=10)
        int_func(times, values, fill_value=0)
        assert_raises(ValueError,
                      int_func, times, values, fill=10, fill_value=9)
        int_func(times, values, fill=np.nan, fill_value=np.nan)
        assert_raises(ValueError,
                      int_func, times, values, fill=10, fill_value=np.nan)
        assert_raises(ValueError,
                      int_func, times, values, fill=np.nan, fill_value=0)


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
    # Name default
    assert_false(re.match(r'step\d+\(t\)$', str(s)) is None)
    # Name reloaded
    s = step_function([0,4,5],[4,2,1], name='goodie_goodie_yum_yum')
    assert_equal(str(s), 'goodie_goodie_yum_yum(t)')


def test_blocks():
    on_off = [[1,2],[3,4]]
    tval = np.array([0.4,1.4,2.4,3.4])
    b = blocks(on_off)
    lam = lambdify(t, b)
    assert_array_equal(lam(tval), [0, 1, 0, 1])
    b = blocks(on_off, amplitudes=[3,5])
    lam = lambdify(t, b)
    assert_array_equal(lam(tval), [0, 3, 0, 5])
    # Check what happens with names
    # Default is from step function
    assert_false(re.match(r'step\d+\(t\)$', str(b)) is None)
    # Can pass in another
    b = blocks(on_off, name='funky_chicken')
    assert_equal(str(b), 'funky_chicken(t)')





def numerical_convolve(func1, func2, interval, dt):
    mni, mxi = interval
    time = np.arange(mni, mxi, dt)
    vec1 = func1(time).astype(float)
    vec2 = func2(time).astype(float)
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
    ff1 = lambdify(t, f1)
    # Time delta
    dt = 1e-3
    # The convolution of ``f1`` with itself is a triangular wave on
    # [0, 2], peaking at 1 with height 1
    tri = convolve_functions(f1, f1, [0, 2], [0, 2], dt, name='conv')
    assert_equal(str(tri), 'conv(t)')
    ftri = lambdify(t, tri)
    time, value = numerical_convolve(ff1, ff1, [0, 2], dt)
    y = ftri(time)
    # numerical convolve about the same as ours
    assert_array_almost_equal(value, y)
    # peak is at 1
    assert_array_almost_equal(time[np.argmax(y)], 1)
    # Flip the interval and get the same result
    for seq1, seq2 in (((0, 2), (2, 0)),
                       ((2, 0), (0, 2)),
                       ((2, 0), (2, 0))):
        tri = convolve_functions(f1, f1, seq1, seq2, dt)
        ftri = lambdify(t, tri)
        y = ftri(time)
        assert_array_almost_equal(value, y)
    # offset square wave by 1 - offset triangle by 1
    f2 = (t > 1) * (t < 2)
    tri = convolve_functions(f1, f2, [0, 3], [0, 3], dt)
    ftri = lambdify(t, tri)
    o1_time = np.arange(0, 3, dt)
    z1s = np.zeros((np.round(1./dt)))
    assert_array_almost_equal(ftri(o1_time), np.r_[z1s, value])
    # Same for input function
    tri = convolve_functions(f2, f1, [0, 3], [0, 3], dt)
    ftri = lambdify(t, tri)
    assert_array_almost_equal(ftri(o1_time), np.r_[z1s, value])
    # 2 seconds for both
    tri = convolve_functions(f2, f2, [0, 4], [0, 4], dt)
    ftri = lambdify(t, tri)
    o2_time = np.arange(0, 4, dt)
    assert_array_almost_equal(ftri(o2_time), np.r_[z1s, z1s, value])
    # offset by -0.5 - offset triangle by -0.5
    f3 = (t > -0.5) * (t < 0.5)
    tri = convolve_functions(f1, f3, [0, 2], [-0.5, 1.5], dt)
    ftri = lambdify(t, tri)
    o1_time = np.arange(-0.5, 1.5, dt)
    assert_array_almost_equal(ftri(o1_time), value)
    # Same for input function
    tri = convolve_functions(f3, f1, [-0.5, 1.5], [0, 2], dt)
    ftri = lambdify(t, tri)
    assert_array_almost_equal(ftri(o1_time), value)
    # -1 second for both
    tri = convolve_functions(f3, f3, [-0.5, 1.5], [-0.5, 1.5], dt)
    ftri = lambdify(t, tri)
    o2_time = np.arange(-1, 1, dt)
    assert_array_almost_equal(ftri(o2_time), value)
    # Check it's OK to be off the dt grid
    tri = convolve_functions(f1, f1, [dt/2, 2 + dt/2], [0, 2], dt, name='conv')
    ftri = lambdify(t, tri)
    assert_array_almost_equal(ftri(time), value, 3)
