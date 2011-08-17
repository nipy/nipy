""" Testing round numbers utility
"""

import numpy as np

from ..doctester import round_numbers, strip_array_repr

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def test_strip_array_repr():
    # check array repr removal
    for arr in (np.array(1),
                np.array(1, dtype=bool),
                np.arange(12),
                np.arange(12).reshape((3,4)),
                np.zeros((3,4), dtype=[('f1', 'f'), ('f2', int)])):
        expected = arr.tolist()
        list_repr = strip_array_repr(repr(arr)).replace('\n', '')
        actual = eval(list_repr)
        assert_equal(expected, actual)


def test_round_numbers():
    # Test string floating point tranformation
    in_out_strs = ( # input, 4DP, 6DP output
        ('100', '100', '100'),
        ('A string', 'A string', 'A string'),
        ('0.25', '0.2500', '0.250000'),
        ('0.12345', '0.1235', '0.123450'), # round up 4DP
        ('0.12343', '0.1234', '0.123430'), # round down 4DP
        ('0.1234567', '0.1235', '0.123457'), # round up 6DP
        ('0.1234564', '0.1235', '0.123456'), # round down 6DP
        ('345.1234564', '345.1235', '345.123456'), # round down 6DP
        ('0.1234564e-10', '0.1235e-10', '0.123456e-10'), # round down 6DP
        ('a0.1234564', 'a0.1234564','a0.1234564'),
        ('0.1234564a', '0.1234564a','0.1234564a'),
        ('_0.1234564', '_0.1234564','_0.1234564'),
        ('0.1234564_', '0.1234564_','0.1234564_'),
        ('(0.1234567)', '(0.1235)', '(0.123457)'), # round up 6DP
        ('(0.1234564)', '(0.1235)', '(0.123456)'), # round down 6DP
        ('(0.1234564e2)', '(0.1235e2)', '(0.123456e2)'), # round down 6DP
        ('(0.1234567)\n{0.7654321}',
         '(0.1235)\n{0.7654}',
         '(0.123457)\n{0.765432}'),
                  )
    for in_str, out_4, out_6 in in_out_strs:
        assert_equal(round_numbers(in_str, 4), out_4)
        assert_equal(round_numbers(in_str, 6), out_6)

