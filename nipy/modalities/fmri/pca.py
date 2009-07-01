"""
This module provides a class for principal components analysis (PCA).

PCA is an orthonormal, linear transform (i.e., a rotation) that maps the
data to a new coordinate system such that the maximal variability of the
data lies on the first coordinate (or the first principal component), the
second greatest variability is projected onto the second coordinate, and
so on.  The resulting data has unit covariance (i.e., it is decorrelated).
This technique can be used to reduce the dimensionality of the data.

More specifically, the data is projected onto the eigenvectors of the
covariance matrix.
"""

__docformat__ = 'restructuredtext'

import numpy as np
import numpy.linalg as L
from nipy.fixes.scipy.stats.models.utils import recipr

def pca(data, mask=None, ncomp=1, standardize=True,
        design_keep=None, design_resid=None):
    """
    Compute the PCA of an image (over ``axis=0``). Image coordmap should
    have a subcoordmap method.

    Parameters
    ----------

    data : ndarray-like (np.float)
        The image on which to perform PCA over its first axis.

    mask : ndarray-like (np.bool)
        An optional mask, should have shape == image.shape[1:]

    ncomp : int
        How many components to return. All the time series
        are returned but only ncomp of the images are computed.

    standardize : bool
        Standardize so each time series has same error-sum-of-squares?

    design_keep : ndarray
        Data is projected onto the column span of design_keep.
        Defaults to np.identity(data.shape[0])

    design_resid : ndarray
        After projecting onto the column span of design_keep, data is
        projected perpendicular to the column span of this matrix.
        Defaults to a matrix of 1s, removing the mean.

    """

    data = np.asarray(data)
    if mask is not None:
        mask = np.asarray(mask)

    if mask is not None:
        nvoxel = mask.sum()
    else:
        nvoxel = np.product(data.shape[1:])

    nimages = data.shape[0]

    if design_keep is not None:
        pinv_design_keep = L.pinv(design_keep)
        def project_keep(Y):
            return np.dot(np.dot(design_keep, pinv_design_keep), Y)
    else:
        def project_keep(Y):
            return Y

    if design_resid is None:
        design_resid = np.ones((data.shape[0], 1))
    pinv_design_resid = L.pinv(design_resid)

    def project_resid(Y):
        return Y - np.dot(np.dot(design_resid, pinv_design_resid), Y)

    """
    Perform the computations needed for the PCA.
    This stores the covariance/correlation matrix of the data in
    the attribute 'C'.
    The components are stored as the attributes 'components', 
    for an fMRI image these are the time series explaining the most
    variance.

    Now, we compute projection matrices. First, data is projected
    onto the columnspace of design_keep, then
    it is projected perpendicular to column space of 
    design_resid.

    """

    if design_keep is None:
        design_keep = np.identity(nimages)

    X = np.dot(design_keep, L.pinv(design_keep))
    XZ = X - np.dot(design_resid, np.dot(L.pinv(design_resid), X))
    UX, SX, VX = L.svd(XZ, full_matrices=0)

    # The matrix UX has orthonormal columns and represents the
    # final "column space" that the data will be projected onto.

    rank = np.greater(SX/SX.max(), 0.01).astype(np.int32).sum()
    UX = UX[:,range(rank)].T

    C = np.zeros((rank, rank))
    for i in range(data.shape[1]):
        Y = data[:,i].reshape((data.shape[0], np.product(data.shape[2:])))
        YX = np.dot(UX, Y)

        if standardize:
            S2 = (project_resid(Y)**2).sum(0)
            Smhalf = recipr(np.sqrt(S2)); del(S2)
            YX *= Smhalf

        if mask is not None:
            YX = YX * np.nan_to_num(mask[i].reshape(Y.shape[1]))

        C += np.dot(YX, YX.T)

    D, Vs = L.eigh(C)
    order = np.argsort(-D)
    D = D[order]
    pcntvar = D * 100 / D.sum()

    time_series = np.dot(UX.T, Vs).T[order]

    """
    Output the component images -- by default, we only output the first
    principal component.

    """

    subVX = time_series[:ncomp]

    out = np.empty((ncomp,) + data.shape[1:], np.float)
    for i in range(data.shape[1]):
        Y = data[:,i].reshape((data.shape[0], np.product(data.shape[2:])))
        U = np.dot(subVX, Y)

        if standardize:
            S2 = (project_resid(Y)**2).sum(0)
            Smhalf = recipr(np.sqrt(S2)); del(S2)
            YX *= Smhalf

        if mask is not None:
            YX *= np.nan_to_num(mask[i].reshape(Y.shape[1]))

        U.shape = (U.shape[0],) + data.shape[2:]
        out[:,i] = U
    return {'time_series':time_series[:ncomp,],
            'pcnt_var': pcntvar,
            'images':out, 
            'rank':rank}






