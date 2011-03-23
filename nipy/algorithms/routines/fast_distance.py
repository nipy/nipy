# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
this module contains a function to perform fast distance computation on arrays

Author : Bertrand Thirion, 2008-2011
"""
import numpy as np


def euclidean_distance(X, Y=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors

    Parameters
    ----------
    X, array of shape (n1,p)
    Y=None, array of shape (n2,p)
            if Y==None, then Y=X is used instead

    Returns
    -------
    ED, array fo shape(n1, n2) with all the pairwise distance
    """
    if Y == None:
        Y = X
    if X.shape[1] != Y.shape[1]:
        raise ValueError("incompatible dimension for X and Y matrices")

    n1 = X.shape[0]
    n2 = Y.shape[0]
    NX = np.reshape(np.sum(X * X, 1), (n1, 1))
    NY = np.reshape(np.sum(Y * Y, 1), (1, n2))
    ED = np.repeat(NX, n2, 1)
    ED += np.repeat(NY, n1, 0)
    ED -= 2 * np.dot(X, Y.T)
    ED = np.maximum(ED, 0)
    ED = np.sqrt(ED)
    return ED
