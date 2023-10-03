
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ....core.image.image_spaces import make_xyz_image
from ....testing import assert_almost_equal
from .._registration import _joint_histogram
from ..affine import Affine, Rigid
from ..histogram_registration import HistogramRegistration

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
    R = HistogramRegistration(I, I, from_bins=clI,
                              from_mask=mask, to_mask=mask)
    R.subsample(spacing=[1, 1, 1])
    Ic = R._from_data
    Ic2 = R._to_data[1:-1, 1:-1, 1:-1]
    assert_array_equal(Ic, Ic2)
    dyn = Ic.max() + 1
    assert dyn == R._joint_hist.shape[0]
    assert dyn == R._joint_hist.shape[1]
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
    J = make_xyz_image(I.get_fdata().copy(), dummy_affine, 'scanner')
    R = HistogramRegistration(I, J)
    R.subsample(spacing=[2, 1, 3])
    R.similarity = simi
    assert_almost_equal(R.eval(Affine()), val)


def _test_renormalization1(simi):
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    R = HistogramRegistration(I, I)
    R.subsample(spacing=[2, 1, 3])
    R._set_similarity(simi, renormalize=True)
    assert R.eval(Affine()) > 1e5


def _test_renormalization2(simi):
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    I0 = make_xyz_image(np.zeros(I.shape, dtype='int16'),
                        dummy_affine, 'scanner')
    R = HistogramRegistration(I0, I)
    R.subsample(spacing=[2, 1, 3])
    R._set_similarity(simi, renormalize=True)
    assert_almost_equal(R.eval(Affine()), 0)


def test_correlation_coefficient():
    _test_similarity_measure('cc', 1.0)


def test_correlation_ratio():
    _test_similarity_measure('cr', 1.0)


def test_correlation_ratio_L1():
    _test_similarity_measure('crl1', 1.0)


def test_supervised_likelihood_ratio():
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    J = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    R = HistogramRegistration(I, J, similarity='slr', dist=np.ones((256, 256)) / (256 ** 2))
    assert_almost_equal(R.eval(Affine()), 0.0)
    pytest.raises(ValueError, HistogramRegistration, I, J, similarity='slr', dist=None)
    pytest.raises(ValueError, HistogramRegistration, I, J, similarity='slr', dist=np.random.rand(100, 127))


def test_normalized_mutual_information():
    _test_similarity_measure('nmi', 1.0)


def test_renormalized_correlation_coefficient():
    _test_renormalization1('cc')
    _test_renormalization2('cc')


def test_renormalized_correlation_ratio():
    _test_renormalization1('cr')
    _test_renormalization2('cr')


def test_renormalized_correlation_ratio_l1():
    _test_renormalization1('crl1')
    _test_renormalization2('crl1')


def test_joint_hist_eval():
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    J = make_xyz_image(I.get_fdata().copy(), dummy_affine, 'scanner')
    # Obviously the data should be the same
    assert_array_equal(I.get_fdata(), J.get_fdata())
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
    vox_coords = np.ascontiguousarray(vox_coords.astype(np.double))
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
    J = make_xyz_image(I.get_fdata().copy(), dummy_affine, 'scanner')
    R = HistogramRegistration(I, J)
    pytest.raises(ValueError, R.subsample, spacing=[0, 1, 3])


def test_set_fov():
    I = make_xyz_image(make_data_int16(), dummy_affine, 'scanner')
    J = make_xyz_image(I.get_fdata().copy(), dummy_affine, 'scanner')
    R = HistogramRegistration(I, J)
    R.set_fov(npoints=np.prod(I.shape))
    assert R._from_data.shape == I.shape
    half_shape = tuple([I.shape[i] / 2 for i in range(3)])
    R.set_fov(spacing=(2, 2, 2))
    assert R._from_data.shape == half_shape
    R.set_fov(corner=half_shape)
    assert R._from_data.shape == half_shape
    R.set_fov(size=half_shape)
    assert R._from_data.shape == half_shape


def test_histogram_masked_registration():
    """ Test the histogram registration class.
    """
    I = make_xyz_image(make_data_int16(dx=100, dy=100, dz=50),
                       dummy_affine, 'scanner')
    J = make_xyz_image(make_data_int16(dx=100, dy=100, dz=50),
                       dummy_affine, 'scanner')
    mask = (np.zeros((100, 100, 50)) == 1)
    mask[10:20, 10:20, 10:20] = True
    R = HistogramRegistration(I, J, to_mask=mask, from_mask=mask)
    sim1 = R.eval(Affine())
    I = make_xyz_image(I.get_fdata()[mask].reshape(10, 10, 10),
                       dummy_affine, 'scanner')
    J = make_xyz_image(J.get_fdata()[mask].reshape(10, 10, 10),
                       dummy_affine, 'scanner')
    R = HistogramRegistration(I, J)
    sim2 = R.eval(Affine())
    assert sim1 == sim2


def test_similarity_derivatives():
    """ Test gradient and Hessian computation of the registration
    objective function.
    """
    I = make_xyz_image(make_data_int16(dx=100, dy=100, dz=50),
                       dummy_affine, 'scanner')
    J = make_xyz_image(np.ones((100, 100, 50), dtype='int16'),
                       dummy_affine, 'scanner')
    R = HistogramRegistration(I, J)
    T = Rigid()
    g = R.eval_gradient(T)
    assert g.dtype == float
    assert_array_equal(g, np.zeros(6))
    H = R.eval_hessian(T)
    assert H.dtype == float
    assert_array_equal(H, np.zeros((6, 6)))


def test_smoothing():
    """ Test smoothing the `to` image.
    """
    I = make_xyz_image(make_data_int16(dx=100, dy=100, dz=50),
                       dummy_affine, 'scanner')
    T = Rigid()
    R = HistogramRegistration(I, I)
    R1 = HistogramRegistration(I, I, smooth=1)
    s = R.eval(T)
    s1 = R1.eval(T)
    assert_almost_equal(s, 1)
    assert s1 < s
    pytest.raises(ValueError, HistogramRegistration, I, I, smooth=-1)
