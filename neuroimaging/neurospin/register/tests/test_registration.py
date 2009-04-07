#!/usr/bin/env python

from neuroimaging.testing import TestCase, assert_equal, assert_almost_equal, \
    assert_raises
import numpy as np
from neuroimaging.neurospin import register 



class Image(object):
    """ 
    Empty object to easily create image objects independently from any I/O package.
    """

    def __init__(self, array, toworld=None, voxsize=[1, 1, 1]):
        self.array = array
        self.voxsize = np.asarray(voxsize)
        if toworld == None: 
            toworld = np.diag(np.concatenate((self.voxsize, [1])))
        self.toworld = toworld



def make_data_uint8(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('uint8')

def make_data_int16(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('int16')

def make_data_float64(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('float64')

def _test_clamping(I, thI=0.0, clI=256):
    IM = register.IconicMatcher(I.array, I.array, I.toworld, I.toworld, thI, thI, bins=clI);                
    Ic = IM.source_clamped
    Ic2 = IM.target_clamped[1:I.array.shape[0]+1,1:I.array.shape[1]+1,1:I.array.shape[2]+1].squeeze()
    assert_equal(Ic, Ic2)
    dyn = Ic.max() + 1
    assert_equal(dyn, IM.joint_hist.shape[0])
    assert_equal(dyn, IM.joint_hist.shape[1])
    assert_equal(dyn, IM.source_hist.shape[0])
    assert_equal(dyn, IM.target_hist.shape[0])

def test_clamping_uint8(): 
    I = Image(make_data_uint8())
    _test_clamping(I)

def test_clamping_uint8_nonstd():
    I = Image(make_data_uint8())
    _test_clamping(I, 10, 165)

def test_clamping_int16(): 
    I = Image(make_data_int16())
    _test_clamping(I)

def test_clamping_int16_nonstd(): 
    I = Image(make_data_int16())
    _test_clamping(I, 10, 165)

def test_clamping_float64(): 
    I = Image(make_data_float64())
    _test_clamping(I)

def test_clamping_float64_nonstd():
    I = Image(make_data_float64())
    _test_clamping(I, 10, 165)
        
def _test_similarity_measure(simi, val):
    I = Image(make_data_int16())
    J = Image(I.array.copy())
    IM = register.IconicMatcher(I.array, J.array, I.toworld, J.toworld)
    IM.set_field_of_view(subsampling=[2,1,3])
    IM.set_similarity(simi)
    assert_almost_equal(IM.eval(np.eye(4)), val)

def test_correlation_coefficient():
    _test_similarity_measure('cc', 1.0) 

def test_correlation_ratio():
    _test_similarity_measure('cr', 1.0) 

def test_normalized_mutual_information():
    _test_similarity_measure('nmi', 1.0) 

def test_explore(): 
    I = Image(make_data_int16())
    J = Image(make_data_int16())
    IM = register.IconicMatcher(I.array, J.array, I.toworld, J.toworld)
    T = np.eye(4)
    T[0:3,3] = np.random.rand(3)
    simi, params = IM.explore(ux=[-1,0,1],uy=[-1,0,1])

def test_iconic():
    """ Test the iconic class.
    """
    I = Image(make_data_int16())
    J = Image(I.array.copy())
    IM = register.IconicMatcher(I.array, J.array, I.toworld, J.toworld)
    assert_raises(ValueError, IM.set_field_of_view, subsampling=[0,1,3])

def _test_resampling(Tv):
    """
    Adding this new test to check whether resample
    may be replaced with scipy.ndimage.affine_transform    
    """
    from neuroimaging.neurospin.register.transform import resample
    from scipy.ndimage import affine_transform
    import time
    I = Image(make_data_int16())
    matrix = Tv[0:3,0:3]
    offset = Tv[0:3,3]
    output_shape = I.array.shape
    t0 = time.clock()
    I1 = resample(I.array, output_shape, Tv)
    dt1 = time.clock()-t0
    t0 = time.clock()
    I2 = affine_transform(I.array, matrix, offset=offset, output_shape=output_shape)
    dt2 = time.clock()-t0
    assert_almost_equal(I1, I2)
    print('3d array resampling')
    print('  using fff: %f sec' % dt1)
    print('  using scipy.ndimage: %f sec' % dt2)


def test_resampling():
    """
    Generate a random similarity transformation
    """
    from neuroimaging.neurospin.register.transform import rotation_vec2mat
    rot = .1*np.random.rand(3) 
    sca = 1+.2*np.random.rand()
    matrix = sca*rotation_vec2mat(rot)
    offset = 10*np.random.rand(3)
    Tv = np.eye(4) 
    Tv[0:3,0:3] = matrix
    Tv[0:3,3] = offset
    """
    DESACTIVATED UNTIL RESAMPLING IS PUT BACK
    _test_resampling(Tv)
    """

if __name__ == "__main__":
        import nose
        nose.run(argv=['', __file__])
