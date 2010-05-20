# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipy.testing import *

import numpy as np

import nipy.core.transforms.affines as affines


def build_xform():
    mat = np.arange(9).reshape((3, 3))
    vec = np.arange(3) + 10
    xform = np.empty((4, 4), dtype=mat.dtype)
    xform[:3, :3] = mat[:]
    xform[3, :] = [0, 0, 0, 1]
    xform[:3, 3] = vec[:]
    return mat, vec, xform


def test_to_matrix_vector():
    mat, vec, xform = build_xform()
    newmat, newvec = affines.to_matrix_vector(xform)
    yield assert_equal, newmat, mat
    yield assert_equal, newvec, vec


def test_from_matrix_vector():
    mat, vec, xform = build_xform()
    newxform = affines.from_matrix_vector(mat, vec)
    assert_equal, newxform, xform
