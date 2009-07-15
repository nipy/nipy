import time

from nipy.testing import assert_almost_equal
import numpy as np
from scipy.ndimage import affine_transform

from nipy.neurospin.register.transform import rotation_vec2mat
from nipy.neurospin.register.routines import cspline_resample


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



def make_data_int16(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('int16')


def resampling(Tv):
    """
    Adding this new test to check whether resample
    may be replaced with scipy.ndimage.affine_transform    
    """
    I = Image(make_data_int16())
    t0 = time.clock()
    I1 = cspline_resample(I.array, I.array.shape, Tv)
    dt1 = time.clock()-t0
    t0 = time.clock()
    I2 = affine_transform(I.array, Tv[0:3,0:3], offset=Tv[0:3,3], output_shape=I.array.shape)
    dt2 = time.clock()-t0
    assert_almost_equal(I1, I2)
    print('3d array resampling')
    print('  using neuroimaging.neurospin: %f sec' % dt1)
    print('  using scipy.ndimage: %f sec' % dt2)


def bench_resampling():
    """
    Generate a random similarity transformation
    """
    rot = .1*np.random.rand(3) 
    sca = 1+.2*np.random.rand()
    matrix = sca*rotation_vec2mat(rot)
    offset = 10*np.random.rand(3)
    Tv = np.eye(4)
    Tv[0:3,0:3] = matrix
    Tv[0:3,3] = offset
    resampling(Tv)
