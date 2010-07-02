# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing fmri utils

"""

import numpy as np

from nipy.modalities.fmri.formula import Term
from nipy.modalities.fmri.aliased import lambdify
from nipy.modalities.fmri.utils import (events,
                                        blocks,
                                        linear_interp,
                                        step_function)
from sympy import Symbol, Function, DiracDelta
import nipy.modalities.fmri.hrf as mfhrf

from nose.tools import (assert_true, assert_false, 
                        assert_equal, assert_raises, raises)

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.testing import parametric

t = Term('t')
TIME_DTYPE = np.dtype([('t', np.float)])

@parametric
def test_events():
    # test events utility function
    h = Function('hrf')
    evs = events([3,6,9])
    yield assert_equal(DiracDelta(-9 + t) + DiracDelta(-6 + t) +
                       DiracDelta(-3 + t), evs)
    evs = events([3,6,9], f=h)
    yield assert_equal(h(-3 + t) + h(-6 + t) + h(-9 + t), evs)
    # test no error for numpy int arrays
    onsets = np.array([30, 70, 100], dtype=np.int64)
    evs = events(onsets, f=mfhrf.glover)


@parametric
def test_linear_inter():
    times = [0,4,5.]
    values = [2.,4,6]
    s = linear_interp(times, values, bounds_error=False)
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
                             [0, 3, 0, 6])
    a = Symbol('a')
    b = blocks(on_off, amplitudes=[3,5], g=a+1)
    lam = lambdify(t, b)
    yield assert_array_equal(lam(tval),
                             [0, 4, 0, 6])

