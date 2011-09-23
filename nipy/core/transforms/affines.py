# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions working on affine transformation matrices.
"""

import numpy as np

def to_matrix_vector(transform):
    """Split a transform into its matrix and vector components.

    The tranformation must be represented in homogeneous coordinates
    and is split into its rotation matrix and translation vector
    components.

    Parameters
    ----------
    transform : array
        NxM transform matrix in homogeneous coordinates representing an
        affine transformation from an (N-1)-dimensional space to an
        (M-1)-dimensional space. An example is a 4x4 transform
        representing rotations and translations in 3 dimensions. A 4x3
        matrix can represent a 2-dimensional plane embedded in 3
        dimensional space.

    Returns
    -------
    matrix, vector : array
        The matrix and vector components of the transform matrix.  For
        an NxM transform, matrix will be N-1xM-1 and vector will be
        1xN-1.

    See Also
    --------
    from_matrix_vector
    """
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return matrix, vector


def from_matrix_vector(matrix, vector):
    """ Combine a matrix and vector into a homogeneous affine

    Combine a rotation matrix and translation vector into a transform
    in homogeneous coordinates.
    
    Parameters
    ----------
    matrix : array
        An NxM array representing the the linear part of the transform.
        A transform from an M-dimensional space to an N-dimensional space.
    vector : array
        A 1xN array representing the translation.

    Returns
    -------
    xform : array
        An N+1xM+1 transform matrix.

    See Also
    --------
    to_matrix_vector
    """
    nin, nout = matrix.shape
    t = np.zeros((nin+1,nout+1), matrix.dtype)
    t[0:nin, 0:nout] = matrix
    t[nin, nout] = 1.
    t[0:nin, nout] = vector
    return t


