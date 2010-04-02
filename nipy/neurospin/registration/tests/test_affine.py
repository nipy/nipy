#!/usr/bin/env python

from nipy.testing import assert_almost_equal
import numpy as np

from nipy.neurospin.registration.affine import Affine, rotation_mat2vec


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
    assert_almost_equal(T.__array__(), np.dot(T1.__array__(), T2.__array__()))

def test_compose(): 
    T1 = Affine(random_vec12('affine'))
    T2 = Affine(random_vec12('similarity'))
    T = T1*T2
    assert_almost_equal(T.__array__(), np.dot(T1.__array__(), T2.__array__()))

def test_mat2vec(): 
    mat = np.eye(4)
    tmp = np.random.rand(3,3)
    U, s, Vt = np.linalg.svd(tmp)
    U /= np.linalg.det(U)
    Vt /= np.linalg.det(Vt)
    mat[0:3,0:3] = np.dot(np.dot(U, np.diag(s)), Vt)
    T = Affine(mat)
    assert_almost_equal(np.asarray(T), mat)


def test_rotation_mat2vec(): 
    r = rotation_mat2vec(np.diag([-1,1,-1]))
    assert(not np.isnan(r).max())
    
