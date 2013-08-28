""" Testing timefuncs module
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


from .. import timefuncs as tf


def test_ascending():
    tr = 2.
    for func in (tf.st_01234, tf.ascending):
        for n_slices in (10, 11):
            assert_almost_equal(
                func(n_slices, tr),
                np.arange(n_slices) / n_slices * tr)
        assert_array_equal(
            np.argsort(func(5, 1)), [0, 1, 2, 3, 4])
        assert_equal(tf.SLICETIME_FUNCTIONS[func.__name__], func)


def test_descending():
    tr = 2.
    for func in (tf.st_43210, tf.descending):
        for n_slices in (10, 11):
            assert_almost_equal(
                func(n_slices, tr),
                np.arange(n_slices-1, -1, -1) / n_slices * tr)
        assert_array_equal(
            np.argsort(func(5, 1)), [4, 3, 2, 1, 0])
        assert_equal(tf.SLICETIME_FUNCTIONS[func.__name__], func)


def test_asc_alt_2():
    tr = 2.
    for func in (tf.st_02413, tf.asc_alt_2):
        assert_almost_equal(
            func(10, tr) / tr * 10,
            [0, 5, 1, 6, 2, 7, 3, 8, 4, 9])
        assert_almost_equal(
            func(11, tr) / tr * 11,
            [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5])
        assert_array_equal(
            np.argsort(func(5, 1)), [0, 2, 4, 1, 3])
        assert_equal(tf.SLICETIME_FUNCTIONS[func.__name__], func)


def test_desc_alt_2():
    tr = 2.
    for func in (tf.st_42031, tf.desc_alt_2):
        assert_almost_equal(
            func(10, tr) / tr * 10,
            [9, 4, 8, 3, 7, 2, 6, 1, 5, 0])
        assert_almost_equal(
            func(11, tr) / tr * 11,
            [5, 10, 4, 9, 3, 8, 2, 7, 1, 6, 0])
        assert_array_equal(
            np.argsort(func(5, 1)), [4, 2, 0, 3, 1])
        assert_equal(tf.SLICETIME_FUNCTIONS[func.__name__], func)


def test_asc_alt_2_1():
    tr = 2.
    for func in (tf.st_13024, tf.asc_alt_2_1):
        assert_almost_equal(
            func(10, tr) / tr * 10,
            [5, 0, 6, 1, 7, 2, 8, 3, 9, 4])
        assert_almost_equal(
            func(11, tr) / tr * 11,
            [5, 0, 6, 1, 7, 2, 8, 3, 9, 4, 10])
        assert_array_equal(
            np.argsort(func(5, 1)), [1, 3, 0, 2, 4])
        assert_equal(tf.SLICETIME_FUNCTIONS[func.__name__], func)


def test_asc_alt_siemens():
    tr = 2.
    for func in (tf.st_odd0_even1, tf.asc_alt_siemens):
        assert_almost_equal(
            func(10, tr) / tr * 10,
            [5, 0, 6, 1, 7, 2, 8, 3, 9, 4])
        assert_almost_equal(
            func(11, tr) / tr * 11,
            [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5])
        assert_array_equal(
            np.argsort(func(5, 1)), [0, 2, 4, 1, 3])
        assert_equal(tf.SLICETIME_FUNCTIONS[func.__name__], func)


def test_asc_alt_half():
    tr = 2.
    for func in (tf.st_03142, tf.asc_alt_half):
        assert_almost_equal(
            func(10, tr) / tr * 10,
            [0, 2, 4, 6, 8, 1, 3, 5, 7, 9])
        assert_almost_equal(
            func(11, tr) / tr * 11,
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9])
        assert_array_equal(
            np.argsort(func(5, 1)), [0, 3, 1, 4, 2])
        assert_equal(tf.SLICETIME_FUNCTIONS[func.__name__], func)


def test_desc_alt_half():
    tr = 2.
    for func in (tf.st_41302, tf.desc_alt_half):
        assert_almost_equal(
            func(10, tr) / tr * 10,
            [9, 7, 5, 3, 1, 8, 6, 4, 2, 0])
        assert_almost_equal(
            func(11, tr) / tr * 11,
            [9, 7, 5, 3, 1, 10, 8, 6, 4, 2, 0])
        assert_array_equal(
            np.argsort(func(5, 1)), [4, 1, 3, 0, 2])
        assert_equal(tf.SLICETIME_FUNCTIONS[func.__name__], func)


def test_number_names():
    for func in (
        tf.st_01234,
        tf.st_43210,
        tf.st_02413,
        tf.st_42031,
        tf.st_13024,
        tf.st_03142,
        tf.st_41302):
        name = func.__name__
        assert_equal(tf.SLICETIME_FUNCTIONS[name], func)
        assert_equal(tf.SLICETIME_FUNCTIONS[name[3:]], func)
