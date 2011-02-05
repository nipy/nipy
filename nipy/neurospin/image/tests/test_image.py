#!/usr/bin/env python

import numpy as np
from nipy.testing import anatfile, assert_equal, assert_almost_equal
from nibabel import load
from nipy.neurospin.image import Image, apply_affine, inverse_affine

I = Image(load(anatfile))

def test_mask(): 
    Imin = I.data.min()
    Imax = I.data.max()
    J = I[np.where(I.data>(Imin+Imax)/2.)]
    assert_equal(J._data.ndim, 1)

def test_extract_block():
    shape = np.array(I.shape)
    start = shape/4
    stop = 3*shape/4
    J = I[[slice(x[0], x[1], 2) for x in zip(start, stop)]]
    assert_equal(J._data.ndim, 3)

def test_get_values(): 
    assert_equal(I.values().mean(), I.data.mean()) 

def test_interpolate_values(): 
    XYZ = (2*np.array(I.shape)*np.random.rand(3, 10).T).T
    vals = I.values(XYZ)
    assert_equal(vals.size, 10)

def test_apply_affine():
    XYZ = (100*(np.random.rand(10,11,12,3)-.5)).astype('int')
    T = np.eye(4)
    T[0:3,0:3] = np.random.rand(3,3)
    T[0:3,3] = 100*(np.random.rand(3)-.5)
    _XYZ = apply_affine(inverse_affine(T), apply_affine(T, XYZ))
    assert_almost_equal(_XYZ, XYZ)

def test_transform(): 
    T = np.eye(4)
    T[0:3,3] = 10*np.random.rand(3)
    J = I.transform(T)
    assert_equal(J.shape, I.shape)

def test_set(): 
    J = I.set(I.values()**2)
    assert_equal(J.shape, I.shape)



if __name__ == "__main__":
        import nose
        nose.run(argv=['', __file__])
