# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing 

"""

import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal


from nipy.fixes.scipy.stats.models.utils import matrix_rank


def test_matrix_rank():
    # Full rank matrix
    yield assert_equal, 4, matrix_rank(np.eye(4))
    I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    yield assert_equal, matrix_rank(I), 3
    # All zeros - zero rank
    yield assert_equal, matrix_rank(np.zeros((4,4))), 0
    # 1 dimension - rank 1 unless all 0
    yield assert_equal, matrix_rank(np.ones((4,))), 1
    yield assert_equal, matrix_rank(np.zeros((4,))), 0
    # accepts array-like
    yield assert_equal, matrix_rank([1]), 1
