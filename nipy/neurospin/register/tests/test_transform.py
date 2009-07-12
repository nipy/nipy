#!/usr/bin/env python

from nipy.testing import TestCase, assert_equal, assert_almost_equal
import numpy as np

from nipy.neurospin.register.transform import Affine, rotation_mat2vec, rotation_vec2mat, vector12


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
    T1 = Affine(vec12=random_vec12('rigid'))
    T2 = Affine(vec12=random_vec12('rigid'))
    T = T1*T2
    assert_almost_equal(T.__array__(), np.dot(T1.__array__(), T2.__array__()))

def test_similarity_compose(): 
    T1 = Affine(vec12=random_vec12('affine'))
    T2 = Affine(vec12=random_vec12('rigid'))
    T = T1*T2
    assert_almost_equal(T.__array__(), np.dot(T1.__array__(), T2.__array__()))

def test_rotation_mat2vec(): 
    r = rotation_mat2vec(np.diag([-1,1,-1]))
    assert(not np.isnan(r).max())
    
