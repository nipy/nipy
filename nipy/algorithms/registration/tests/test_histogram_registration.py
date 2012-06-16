#!/usr/bin/env python

import numpy as np

from ....core.image.image_spaces import make_xyz_image
from ..affine import Affine
from ..histogram_registration import HistogramRegistration
from .._registration import _joint_histogram

from numpy.testing import assert_array_equal
from ....testing import assert_equal, assert_almost_equal, assert_raises

dummy_affine = np.eye(4)

def make_data_bool(dx=100, dy=100, dz=50):
    return (np.random.rand(dx, dy, dz)
                   - np.random.rand()) > 0

def make_data_uint8(dx=100, dy=100, dz=50):
    return (256 * (np.random.rand(dx, dy, dz)
                   - np.random.rand())).astype('uint8')


def make_data_int16(dx=100, dy=100, dz=50):
    return (256 * (np.random.rand(dx, dy, dz)
                   - np.random.rand())).astype('int16')


def make_data_float64(dx=100, dy=100, dz=50):
    return (256 * (np.random.rand(dx, dy, dz)
                 - np.random.rand())).astype('float64')


def _test_clamping(I, thI=0.0, clI=256, mask=None):
    R = HistogramRegistration(I, I, from_bins=clI, from_mask=mask, to_mask=mask)
    R.subsample(spacing=[1, 1, 1])
    Ic = R._from_data
    Ic2 = R._to_data[1:-1, 1:-1, 1:-1]
    assert_equal(Ic, Ic2)
    dyn = Ic.max() + 1
    assert_equal(dyn, R._joint_hist.shape[0])
    assert_equal(dyn, R._joint_hist.shape[1])
    return Ic, Ic2


def test_clamping_uint8():
    I = make_xyz_image(make_data_uint8(), dummy_affine, 'scanner')
    _test_clamping(I)


def test_clamping_uint8_nonstd():
    I = make_xyz_image(make_data_uint8(), dummy_affine, 'scanner')
    _test_clamping(I, 10, 165)


def test_clamping_int16():
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    _test_clamping(I)


def test_masked_clamping_int16():
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    _test_clamping(I, mask=make_data_bool())


def test_clamping_int16_nonstd():
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    _test_clamping(I, 10, 165)


def test_clamping_float64():
    I = make_xyz_image(make_data_float64(), dummy_affine, 'scanner')
    _test_clamping(I)


def test_clamping_float64_nonstd():
    I = make_xyz_image(make_data_float64(), dummy_affine, 'scanner')
    _test_clamping(I, 10, 165)


def _test_similarity_measure(simi, val):
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    J = make_xyz_image(I.get_data().copy(), dummy_affine, 'scanner')
    R = HistogramRegistration(I, J)
    R.subsample(spacing=[2, 1, 3])
    R.similarity = simi
    assert_almost_equal(R.eval(Affine()), val)


def test_correlation_coefficient():
    _test_similarity_measure('cc', 1.0)


def test_correlation_ratio():
    _test_similarity_measure('cr', 1.0)


def test_correlation_ratio_L1():
    _test_similarity_measure('crl1', 1.0)


def test_normalized_mutual_information():
    _test_similarity_measure('nmi', 1.0)


def test_joint_hist_eval():
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    J = make_xyz_image(I.get_data().copy(), dummy_affine, 'scanner')
    # Obviously the data should be the same
    assert_array_equal(I.get_data(), J.get_data())
    # Instantiate default thing
    R = HistogramRegistration(I, J)
    R.similarity = 'cc'
    null_affine = Affine()
    val = R.eval(null_affine)
    assert_almost_equal(val, 1.0)
    # Try with what should be identity
    R.subsample(spacing=[1, 1, 1])
    assert_array_equal(R._from_data.shape, I.shape)
    val = R.eval(null_affine)
    assert_almost_equal(val, 1.0)


def test_joint_hist_raw():
    # Set up call to joint histogram
    jh_arr = np.zeros((10, 10), dtype=np.double)
    data_shape = (2, 3, 4)
    data = np.random.randint(size=data_shape,
                             low=0, high=10).astype(np.short)
    data2 = np.zeros(np.array(data_shape) + 2, dtype=np.short)
    data2[:] = -1
    data2[1:-1, 1:-1, 1:-1] = data.copy()
    vox_coords = np.indices(data_shape).transpose((1, 2, 3, 0))
    vox_coords = vox_coords.astype(np.double)
    _joint_histogram(jh_arr, data.flat, data2, vox_coords, 0)
    assert_almost_equal(np.diag(np.diag(jh_arr)), jh_arr)


def test_explore():
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    J = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    R = HistogramRegistration(I, J)
    T = Affine()
    simi, params = R.explore(T, (0, [-1, 0, 1]), (1, [-1, 0, 1]))


def test_histogram_registration():
    """ Test the histogram registration class.
    """
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    J = make_xyz_image(I.get_data().copy(), dummy_affine, 'scanner')
    R = HistogramRegistration(I, J)
    assert_raises(ValueError, R.subsample, spacing=[0, 1, 3])


def test_histogram_masked_registration():
    """ Test the histogram registration class.
    """
    I = make_xyz_image(make_data_int16(dx=100, dy=100, dz=50), dummy_affine, 'scanner')
    J = make_xyz_image(make_data_int16(dx=100, dy=100, dz=50), dummy_affine, 'scanner')
    mask = (np.zeros((100,100,50)) == 1)
    mask[10:20,10:20,10:20] = True
    R = HistogramRegistration(I, J, to_mask=mask, from_mask=mask)
    sim1 = R.eval(Affine())
    I = make_xyz_image(I.get_data()[mask].reshape(10,10,10), dummy_affine, 'scanner')
    J = make_xyz_image(J.get_data()[mask].reshape(10,10,10), dummy_affine, 'scanner')
    R = HistogramRegistration(I, J)
    sim2 = R.eval(Affine())
    assert_equal(sim1, sim2)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
