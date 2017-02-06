""" Testing realfuncs module
"""

from os.path import dirname, join as pjoin
from itertools import product

import numpy as np

from ..realfuncs import dct_ii_basis, dct_ii_cut_basis

from nose.tools import assert_raises

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)


HERE = dirname(__file__)


def test_dct_ii_basis():
    # Test DCT-II basis
    for N in (5, 10, 100):
        spm_fname = pjoin(HERE, 'dct_{0}.txt'.format(N))
        spm_mtx = np.loadtxt(spm_fname)
        vol_times = np.arange(N) * 15. + 3.2
        our_dct = dct_ii_basis(vol_times)
        # Check dot products of columns
        sq_col_lengths = np.ones(N) * N / 2.
        sq_col_lengths[0] = N
        assert_almost_equal(our_dct.T.dot(our_dct),
                            np.diag(sq_col_lengths))
        col_lengths = np.sqrt(sq_col_lengths)
        assert_almost_equal(our_dct / col_lengths, spm_mtx)
        # Normalize length
        our_normed_dct = dct_ii_basis(vol_times, normcols=True)
        assert_almost_equal(our_normed_dct, spm_mtx)
        assert_almost_equal(our_normed_dct.T.dot(our_normed_dct), np.eye(N))
        for i in range(N):
            assert_almost_equal(dct_ii_basis(vol_times, i) / col_lengths[:i],
                                spm_mtx[:, :i])
            assert_almost_equal(dct_ii_basis(vol_times, i, True),
                                spm_mtx[:, :i])
    vol_times[0] += 0.1
    assert_raises(ValueError, dct_ii_basis, vol_times)


def test_dct_ii_cut_basis():
    # DCT-II basis with cut frequency
    for dt, cut_period, N in product((0.1, 1.1),
                                     (10.1, 20.1),
                                     (20, 100, 1000)):
        times = np.arange(N) * dt
        order = int(np.floor(2 * N * 1./ cut_period * dt))
        dct_vals = dct_ii_cut_basis(times, cut_period)
        if order == 0:
            assert_array_equal(dct_vals, np.ones((N, 1)))
            continue
        dct_expected = np.ones((N, order))
        dct_expected[:, :-1] = dct_ii_basis(times, order, normcols=True)[:, 1:]
        assert_array_equal(dct_vals, dct_expected)
