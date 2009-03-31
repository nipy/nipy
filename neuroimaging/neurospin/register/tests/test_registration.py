#!/usr/bin/env python

from numpy.testing import TestCase, assert_equal, assert_almost_equal, \
    assert_raises
import numpy as np
from neuroimaging.neurospin import registration 

class Image(object):
    """ 
    Empty object to easily create image objects independently from any I/O package.
    """

    def __init__(self, array, transform=None, voxsize=[1, 1, 1]):
        self.array = array
        self.voxsize = np.asarray(voxsize)
        if transform == None: 
            transform = np.diag(np.concatenate((self.voxsize, [1])))
        self.transform = transform



def make_data_uint8(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('uint8')

def make_data_int16(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('int16')

def make_data_float64(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('float64')

def _test_imatch_same(I, thI=0.0, clI=256):
    iMatch = registration._iconic.imatch(I, I, thI, thI, clI, clI);                
    Ic = iMatch[0]
    Ic2 = iMatch[1][1:I.shape[0]+1,1:I.shape[1]+1,1:I.shape[2]+1].squeeze()
    assert_equal(Ic, Ic2)
    dyn = Ic.max() + 1
    assert_equal(dyn, iMatch[2].shape[0])
    assert_equal(dyn, iMatch[2].shape[1])
    assert_equal(dyn, iMatch[3].shape[0])
    assert_equal(dyn, iMatch[4].shape[0])

def test_imatch_same_uint8_1(): 
    I = make_data_uint8()
    _test_imatch_same(I)

def test_imatch_same_uint8_2():
    I = make_data_uint8()
    _test_imatch_same(I, 10, 165)

def test_imatch_same_int16_1(): 
    I = make_data_int16()
    _test_imatch_same(I)

def test_imatch_same_int16_2(): 
    I = make_data_int16()
    _test_imatch_same(I, 10, 165)

def test_imatch_same_float64_1(): 
    I = make_data_float64()
    _test_imatch_same(I)

def test_imatch_same_float64_2():
    I = make_data_float64()
    _test_imatch_same(I, 10, 165)
        

def test_correlation_coefficient():
    I = Image(make_data_int16())
    J = Image(I.array.copy())
    IM = registration.iconic(I, J)
    IM.set(subsampling=[2,1,3], similarity='correlation coefficient')
    assert_almost_equal(IM.eval(np.eye(4)), 1.0)

def test_correlation_ratio():
    I = Image(make_data_int16())
    J = Image(I.array.copy())
    IM = registration.iconic(I, J)
    IM.set(subsampling=[2,1,3], similarity='correlation ratio')
    assert_almost_equal(IM.eval(np.eye(4)), 1.0)

def test_normalized_mutual_information():
    I = Image(make_data_int16())
    J = Image(I.array.copy())
    IM = registration.iconic(I, J)
    IM.set(subsampling=[2,1,3], similarity='normalized mutual information')
    assert_almost_equal(IM.eval(np.eye(4)), 1.0)

def test_explore(): 
    I = Image(make_data_int16())
    J = Image(make_data_int16())
    IM = registration.iconic(I, J)
    T = np.eye(4)
    T[0:3,3] = np.random.rand(3)
    simi, params = IM.explore(ux=[-1,0,1],uy=[-1,0,1])

def test_iconic():
    """ Test the iconic class.
    """
    I = Image(make_data_int16())
    J = Image(I.array.copy())
    IM = registration.iconic(I, J)
    assert_raises(ValueError, IM.set, subsampling=[0,1,3])

def _test_resampling(Tv):
    """
    Adding this new test to check whether resample
    may be replaced with scipy.ndimage.affine_transform    
    """
    from neuroimaging.neurospin.registration.transform_affine import resample
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
    from neuroimaging.neurospin.registration.transform_affine import rvector_to_matrix
    rot = .1*np.random.rand(3) 
    sca = 1+.2*np.random.rand()
    matrix = sca*rvector_to_matrix(rot)
    offset = 10*np.random.rand(3)
    Tv = np.eye(4) 
    Tv[0:3,0:3] = matrix
    Tv[0:3,3] = offset
    _test_resampling(Tv)


if __name__ == "__main__":
        import nose
        nose.run(argv=['', __file__])
