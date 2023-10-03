# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This test can only be run from the directory above, as it uses relative
imports.
"""

import numpy as np

from ..affine_utils import from_matrix_vector, to_matrix_vector


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
    newmat, newvec = to_matrix_vector(xform)
    np.testing.assert_array_equal(newmat, mat)
    np.testing.assert_array_equal(newvec, vec)


def test_from_matrix_vector():
    mat, vec, xform = build_xform()
    newxform = from_matrix_vector(mat, vec)
    np.testing.assert_array_equal(newxform, xform)
