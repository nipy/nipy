# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing fmri utils

"""

import numpy as np

from nipy.modalities.fmri.formula import Term
from nipy.modalities.fmri.aliased import lambdify
from nipy.modalities.fmri.utils import (events,
                                        blocks,
                                        interp,
                                        linear_interp,
                                        step_function,
                                        convolve_functions,
                                        )
from sympy import Symbol, Function, DiracDelta
import nipy.modalities.fmri.hrf as mfhrf

from nose.tools import (assert_true, assert_false, 
                        assert_equal, assert_raises, raises)

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.testing import parametric

t = Term('t')


@parametric
def test_events():
    # test events utility function
    h = Function('hrf')
    evs = events([3,6,9])
    yield assert_equal(DiracDelta(-9 + t) + DiracDelta(-6 + t) +
                       DiracDelta(-3 + t), evs)
    evs = events([3,6,9], f=h)
    yield assert_equal(h(-3 + t) + h(-6 + t) + h(-9 + t), evs)
    # make some beta symbols
    b = [Symbol('b%d' % i, dummy=True) for i in range(3)]
    a = Symbol('a')
    p = b[0] + b[1]*a + b[2]*a**2
    evs = events([3,6,9], amplitudes=[2,1,-1], g=p)
    yield assert_equal((2*b[1] + 4*b[2] + b[0])*DiracDelta(-3 + t) +
                       (-b[1] + b[0] + b[2])*DiracDelta(-9 + t) +
                       (b[0] + b[1] + b[2])*DiracDelta(-6 + t),
                       evs)
    evs = events([3,6,9], amplitudes=[2,1,-1], g=p, f=h)
    yield assert_equal((2*b[1] + 4*b[2] + b[0])*h(-3 + t) +
                       (-b[1] + b[0] + b[2])*h(-9 + t) +
                       (b[0] + b[1] + b[2])*h(-6 + t),
                       evs)
    # test no error for numpy int arrays
    onsets = np.array([30, 70, 100], dtype=np.int64)
    evs = events(onsets, f=mfhrf.glover)


@parametric
def test_interp():
    times = [0,4,5.]
    values = [2.,4,6]
    for int_func in (interp, linear_interp):
        s = int_func(times, values, bounds_error=False)
        tval = np.array([-0.1,0.1,3.9,4.1,5.1])
        res = lambdify(t, s)(tval)
        yield assert_array_equal(np.isnan(res),
                                 [True, False, False, False, True])
        yield assert_array_almost_equal(res[1:-1],
                                        [2.05, 3.95, 4.2])
        # specifying kind as linear is OK
        s = linear_interp(times, values, kind='linear')


@raises(ValueError)
def test_linear_inter_kind():
    s = linear_interp([0, 1], [1, 2], kind='cubic')


@parametric
def test_step_function():
    # test step function
    # step function is a function of t
    s = step_function([0,4,5],[2,4,6])
    tval = np.array([-0.1,0,3.9,4,4.1,5.1])
    lam = lambdify(t, s)
    yield assert_array_equal(lam(tval), [0, 2, 2, 4, 4, 6])
    s = step_function([0,4,5],[4,2,1])
    lam = lambdify(t, s)
    yield assert_array_equal(lam(tval), [0, 4, 4, 2, 2, 1])
    
    

@parametric
def test_blocks():
    on_off = [[1,2],[3,4]]
    tval = np.array([0.4,1.4,2.4,3.4])
    b = blocks(on_off)
    lam = lambdify(t, b)
    yield assert_array_equal(lam(tval),
                             [0, 1, 0, 1])
    b = blocks(on_off, amplitudes=[3,5])
    lam = lambdify(t, b)
    yield assert_array_equal(lam(tval),
                             [0, 3, 0, 5])


@parametric
def test_convolve_functions():
    # square wave
    expr = (t > 0) * (t < 1)
    # convolve with 1
    cf = convolve_functions(expr, 1, [-1, 2], 0.1)
    lam = lambdify(t, cf)
