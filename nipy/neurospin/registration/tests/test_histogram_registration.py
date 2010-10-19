#!/usr/bin/env python

from nipy.testing import assert_equal, assert_almost_equal, assert_raises
import numpy as np

from nipy.core.image.affine_image import AffineImage 
from nipy.neurospin.registration import HistogramRegistration, Affine

dummy_affine = np.eye(4)

def make_data_uint8(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('uint8')

def make_data_int16(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('int16')

def make_data_float64(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('float64')

def _test_clamping(I, thI=0.0, clI=256):
    R = HistogramRegistration(I, I, from_bins=clI)
    Ic = R._source
    Ic2 = R._target[1:I.shape[0]+1,1:I.shape[1]+1,1:I.shape[2]+1]
    assert_equal(Ic, Ic2.squeeze())
    dyn = Ic.max() + 1
    assert_equal(dyn, R._joint_hist.shape[0])
    assert_equal(dyn, R._joint_hist.shape[1])
    assert_equal(dyn, R._source_hist.shape[0])
    assert_equal(dyn, R._target_hist.shape[0])
    return Ic, Ic2

def test_clamping_uint8(): 
    I = AffineImage(make_data_uint8(), dummy_affine, 'ijk')
    _test_clamping(I)

def test_clamping_uint8_nonstd():
    I = AffineImage(make_data_uint8(), dummy_affine, 'ijk')
    _test_clamping(I, 10, 165)

def test_clamping_int16(): 
    I = AffineImage(make_data_int16(), dummy_affine, 'ijk')
    _test_clamping(I)

def test_clamping_int16_nonstd(): 
    I = AffineImage(make_data_int16(), dummy_affine, 'ijk')
    _test_clamping(I, 10, 165)

def test_clamping_float64(): 
    I = AffineImage(make_data_float64(), dummy_affine, 'ijk')
    _test_clamping(I)

def test_clamping_float64_nonstd():
    I = AffineImage(make_data_float64(), dummy_affine, 'ijk')
    _test_clamping(I, 10, 165)
        
def _test_similarity_measure(simi, val):
    I = AffineImage(make_data_int16(), dummy_affine, 'ijk')
    J = AffineImage(I.get_data().copy(), dummy_affine, 'ijk')
    R = HistogramRegistration(I, J)
    R.set_source_fov(spacing=[2,1,3])
    R.similarity = simi
    assert_almost_equal(R.eval(np.eye(4)), val)

def test_correlation_coefficient():
    _test_similarity_measure('cc', 1.0) 

def test_correlation_ratio():
    _test_similarity_measure('cr', 1.0) 

def test_normalized_mutual_information():
    _test_similarity_measure('nmi', 1.0) 

def test_explore(): 
    I = AffineImage(make_data_int16(), dummy_affine, 'ijk')
    J = AffineImage(make_data_int16(), dummy_affine, 'ijk')
    R = HistogramRegistration(I, J)
    T = Affine()
    simi, params = R.explore(T, (0,[-1,0,1]),(1,[-1,0,1]))

def test_iconic():
    """ Test the iconic registration class.
    """
    I = AffineImage(make_data_int16(), dummy_affine, 'ijk')
    J = AffineImage(I.get_data().copy(), dummy_affine, 'ijk')
    R = HistogramRegistration(I, J)
    assert_raises(ValueError, R.set_source_fov, spacing=[0,1,3])

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
