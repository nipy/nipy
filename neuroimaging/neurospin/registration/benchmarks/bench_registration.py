import time

import numpy as np
from scipy.ndimage import affine_transform

from nipy.neurospin import registration 
from nipy.neurospin.registration.transform_affine import (
    rvector_to_matrix, resample)

class Image(object):
    """Empty object to easily create image objects independently from
    any I/O package.
    """

    def __init__(self, array, transform=None, voxsize=[1, 1, 1]):
        self.array = array
        self.voxsize = np.asarray(voxsize)
        if transform == None: 
            transform = np.diag(np.concatenate((self.voxsize, [1])))
        self.transform = transform


def make_data_int16(dx=100, dy=100, dz=50):
    return (256*(np.random.rand(dx, dy, dz) - np.random.rand())).astype('int16')


def _resampling(Tv):
    """
    Adding this new test to check whether resample
    may be replaced with scipy.ndimage.affine_transform    
    """
    I = Image(make_data_int16())
    matrix = Tv[0:3,0:3]
    offset = Tv[0:3,3]
    output_shape = I.array.shape
    t0 = time.clock()
    I1 = resample(I.array, output_shape, Tv)
    dt1 = time.clock()-t0
    t0 = time.clock()
    I2 = affine_transform(I.array, matrix, offset=offset, 
                          output_shape=output_shape)
    dt2 = time.clock()-t0
    print('3d array resampling')
    print('  using fff: %f sec' % dt1)
    print('  using scipy.ndimage: %f sec' % dt2)


def bench_resampling():
    # Generate a random similarity transformation

    rot = .1*np.random.rand(3) 
    sca = 1+.2*np.random.rand()
    matrix = sca*rvector_to_matrix(rot)
    offset = 10*np.random.rand(3)
    Tv = np.eye(4) 
    Tv[0:3,0:3] = matrix
    Tv[0:3,3] = offset
    _resampling(Tv)
