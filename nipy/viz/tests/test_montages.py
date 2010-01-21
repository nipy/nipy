""" Testing montage module

"""

import numpy as np

from nipy.viz.montages import array_montage, MontageError

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.testing import parametric


@parametric
def test_array_montages():
    arr = np.arange(24).reshape((2,3,4))
    arrz = np.rollaxis(np.rot90(arr), 2)
    blank_slice = np.zeros((3,2)) # refecting rotation
    em_2r = np.vstack([np.hstack(arrz[0:2]), np.hstack(arrz[2:])])
    montage = array_montage(arr)
    yield assert_array_equal(montage, em_2r)
    em_1r = np.hstack(arrz)
    # columns argument sets columns manually
    montage = array_montage(arr, n_columns=4)
    yield assert_array_equal(montage, em_1r)
    montage = array_montage(arr, slice(None))
    yield assert_array_equal(montage, em_2r)
    montage = array_montage(arr, n_columns=2)
    yield assert_array_equal(montage, em_2r)
    montage = array_montage(arr, (0,), n_columns=2) # padded out with blanks
    yield assert_array_equal(montage, np.hstack([arrz[0]] + [blank_slice]))
    # now test running over first axis instead of last
    montage = array_montage(np.rollaxis(arr,2), axis=0)
    yield assert_array_equal(montage, em_2r)
    # not 3D - error
    yield assert_raises(MontageError, array_montage, np.zeros((2,2)))
    yield assert_raises(MontageError, array_montage, np.zeros((2,2,2,2)))
