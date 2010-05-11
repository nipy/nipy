#!/usr/bin/env python

from nipy.testing import assert_equal, assert_almost_equal, assert_raises
import numpy as np

from nipy.neurospin.image import Image 
from nipy.neurospin.registration import IconicRegistration, Affine

dummy_affine = np.eye(4)

def make_data_uint8(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('uint8')

def make_data_int16(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('int16')

def make_data_float64(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('float64')

def _test_clamping(I, thI=0.0, clI=256):
    regie = IconicRegistration(I, I, bins=clI)
    Ic = regie._source
    Ic2 = regie._target[1:I.shape[0]+1,1:I.shape[1]+1,1:I.shape[2]+1]
    assert_equal(Ic, Ic2.squeeze())
    dyn = Ic.max() + 1
    assert_equal(dyn, regie._joint_hist.shape[0])
    assert_equal(dyn, regie._joint_hist.shape[1])
    assert_equal(dyn, regie._source_hist.shape[0])
    assert_equal(dyn, regie._target_hist.shape[0])
    return Ic, Ic2

def test_clamping_uint8(): 
    I = Image(make_data_uint8(), dummy_affine)
    _test_clamping(I)

def test_clamping_uint8_nonstd():
    I = Image(make_data_uint8(), dummy_affine)
    _test_clamping(I, 10, 165)

def test_clamping_int16(): 
    I = Image(make_data_int16(), dummy_affine)
    _test_clamping(I)

def test_clamping_int16_nonstd(): 
    I = Image(make_data_int16(), dummy_affine)
    _test_clamping(I, 10, 165)

def test_clamping_float64(): 
    I = Image(make_data_float64(), dummy_affine)
    _test_clamping(I)

def test_clamping_float64_nonstd():
    I = Image(make_data_float64(), dummy_affine)
    _test_clamping(I, 10, 165)
        
def _test_similarity_measure(simi, val):
    I = Image(make_data_int16(), dummy_affine)
    J = Image(I.data.copy(), dummy_affine)
    regie = IconicRegistration(I, J)
    regie.set_source_fov(spacing=[2,1,3])
    regie.similarity = simi
    assert_almost_equal(regie.eval(np.eye(4)), val)

def test_correlation_coefficient():
    _test_similarity_measure('cc', 1.0) 

def test_correlation_ratio():
    _test_similarity_measure('cr', 1.0) 

def test_normalized_mutual_information():
    _test_similarity_measure('nmi', 1.0) 

def test_explore(): 
    I = Image(make_data_int16(), dummy_affine)
    J = Image(make_data_int16(), dummy_affine)
    regie = IconicRegistration(I, J)
    T = Affine()
    simi, params = regie.explore(T, (0,[-1,0,1]),(1,[-1,0,1]))

def test_iconic():
    """ Test the iconic registration class.
    """
    I = Image(make_data_int16(), dummy_affine)
    J = Image(I.data.copy(), dummy_affine)
    regie = IconicRegistration(I, J)
    assert_raises(ValueError, regie.set_source_fov, spacing=[0,1,3])

if __name__ == "__main__":
        import nose
        nose.run(argv=['', __file__])
