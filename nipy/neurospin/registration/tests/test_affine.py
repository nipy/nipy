#!/usr/bin/env python

import numpy as np

from ..affine import Affine, rotation_mat2vec, apply_affine

from nose.tools import assert_true, assert_false
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nipy.testing import assert_almost_equal


def random_vec12(subtype='affine'): 
    v = np.array([0,0,0,0.0,0,0,1,1,1,0,0,0])
    v[0:3] = 20*np.random.rand(3)
    v[3:6] = np.random.rand(3)
    if subtype == 'similarity': 
        v[6:9] = np.random.rand()
    elif subtype == 'affine': 
        v[6:9] = np.random.rand(3)
        v[9:12] = np.random.rand(3)
    return v


def test_rigid_compose(): 
    T1 = Affine(random_vec12('rigid'))
    T2 = Affine(random_vec12('rigid'))
    T = T1*T2
    assert_almost_equal(T.as_affine(), np.dot(T1.as_affine(), T2.as_affine()))


def test_compose(): 
    T1 = Affine(random_vec12('affine'))
    T2 = Affine(random_vec12('similarity'))
    T = T1*T2
    assert_almost_equal(T.as_affine(), np.dot(T1.as_affine(), T2.as_affine()))


def test_mat2vec(): 
    mat = np.eye(4)
    tmp = np.random.rand(3,3)
    U, s, Vt = np.linalg.svd(tmp)
    U /= np.linalg.det(U)
    Vt /= np.linalg.det(Vt)
    mat[0:3,0:3] = np.dot(np.dot(U, np.diag(s)), Vt)
    T = Affine(mat)
    assert_almost_equal(T.as_affine(), mat)


def test_rotation_mat2vec(): 
    r = rotation_mat2vec(np.diag([-1,1,-1]))
    assert_false(np.isnan(r).max())


def validated_apply_affine(T, xyz):
    # This was the original apply_affine implementation that we've stashed here
    # to test against
    xyz = np.asarray(xyz)
    shape = xyz.shape[0:-1]
    XYZ = np.dot(np.reshape(xyz, (np.prod(shape), 3)), T[0:3,0:3].T)
    XYZ[:,0] += T[0,3]
    XYZ[:,1] += T[1,3]
    XYZ[:,2] += T[2,3]
    XYZ = np.reshape(XYZ, shape+(3,))
    return XYZ


def test_apply_affine():
    aff = np.diag([2, 3, 4, 1])
    pts = np.random.uniform(size=(4,3))
    assert_array_equal(apply_affine(aff, pts),
                       pts * [[2, 3, 4]])
    aff[:3,3] = [10, 11, 12]
    assert_array_equal(apply_affine(aff, pts),
                       pts * [[2, 3, 4]] + [[10, 11, 12]])
    aff[:3,:] = np.random.normal(size=(3,4))
    exp_res = np.concatenate((pts.T, np.ones((1,4))), axis=0)
    exp_res = np.dot(aff, exp_res)[:3,:].T
    assert_array_equal(apply_affine(aff, pts), exp_res)
    # Check we get the same result as the previous implementation
    assert_almost_equal(validated_apply_affine(aff, pts), apply_affine(aff, pts))


def test_compsed_affines():
    aff1 = np.diag([2, 3, 4, 1])
    aff2 = np.eye(4)
    aff2[:3,3] = (10, 11, 12)
    comped = np.dot(aff2, aff1)
    comped_obj = Affine(comped)
    assert_array_almost_equal(comped_obj.as_affine(), comped)
    aff1_obj = Affine(aff1)
    aff2_obj = Affine(aff2)
    re_comped = aff2_obj.compose(aff1_obj)
    assert_array_almost_equal(re_comped.as_affine(), comped)
    # Crazy, crazy, crazy
    aff1_remixed = aff1_obj.as_affine()
    aff2_remixed = aff2_obj.as_affine()
    comped_remixed = np.dot(aff2_remixed, aff1_remixed)
    assert_array_almost_equal(comped_remixed,
                              Affine(comped_remixed).as_affine())
