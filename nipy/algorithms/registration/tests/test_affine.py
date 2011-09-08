#!/usr/bin/env python

import numpy as np

from ..affine import (Affine, Affine2D, Rigid, Rigid2D,
                      Similarity, Similarity2D,
                      rotation_mat2vec)

from nose.tools import assert_true, assert_false
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ....testing import assert_almost_equal


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


"""
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
"""


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


def test_composed_affines():
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


def test_affine_types():
    pts = np.random.normal(size=(10,3))
    for klass, n_params in ((Affine, 12),
                            (Affine2D, 6),
                            (Rigid, 6),
                            (Rigid2D, 3),
                            (Similarity, 7),
                            (Similarity2D, 4),
                           ):
        obj = klass()
        assert_array_equal(obj.param, np.zeros((n_params,)))
        obj.param = np.ones((n_params,))
        assert_array_equal(obj.param, np.ones((n_params,)))
        # Check that round trip works
        orig_aff = obj.as_affine()
        obj2 = klass(orig_aff)
        assert_array_almost_equal(obj2.as_affine(), orig_aff)
        # Check inverse
        inv_obj = obj.inv()
        # Check points transform and invert
        pts_dash = obj.apply(pts)
        assert_array_almost_equal(pts, inv_obj.apply(pts_dash))
        # Check composition with inverse gives identity
        with_inv = inv_obj.compose(obj)
        assert_array_almost_equal(with_inv.as_affine(), np.eye(4))
        # Just check that str works without error
        s = str(obj)


def test_indirect_affines(): 
    T = np.eye(4)
    A = np.random.rand(3,3)
    if np.linalg.det(A) > 0: 
        A = -A
    T[:3,:3] = A
    obj = Affine(T) 
    assert_false(obj.is_direct)
    assert_array_almost_equal(T, obj.as_affine())
    
