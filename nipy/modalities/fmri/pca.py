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

# Local imports

from nipy.core.image.image import Image, rollaxis as image_rollaxis
from nipy.core.image.xyz_image import XYZImage

def pca(xyz_image, mask=None, ncomp=1, axis='t', standardize=True,
        design_keep=None, design_resid=None, 
        ):
    """
    Compute the PCA of an image over a specified axis. 

    Parameters
    ----------

    data : XYZImage
        The image on which to perform PCA over its first axis.

    mask : XYZImage
        An optional mask, should have shape == image.shape[:3]
        and the same XYZTransform.

    ncomp : int
        How many components to return. All the time series
        are returned but only ncomp of the images are computed.

    axis : str or int
        Axis over which to perform PCA. Cannot be a spatial axis
        because the results have to be XYZImages.

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

    if axis in xyz_image.world.coord_names + \
            xyz_image.axes.coord_names[:3] + tuple(range(3)):
        raise ValueError('cannot perform PCA over a spatial axis' +
                         'or we will not be able to output XYZImages')

    xyz_data = xyz_image.get_data()
    image = Image(xyz_data, xyz_image.coordmap)
    image = image_rollaxis(image, axis)

    if mask is not None:
        if mask.xyz_transform != xyz_image.xyz_transform:
            raise ValueError('mask and xyz_image have different coordinate systems')

        if mask.ndim != image.ndim - 1:
            raise ValueError('mask should have one less dimension then xyz_image')

        if mask.axes.coord_names != image.axes.coord_names[1:]:
            raise ValueError('mask should have axes %s' % str(image.axes.coord_names[1:]))
                         
    data = image.get_data()

    if mask is not None:
        mask_data = mask.get_data()

    if mask is not None:
        nvoxel = mask_data.sum()
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
            Smhalf = np.nan_to_num(1./np.sqrt(S2)); del(S2)
            YX *= Smhalf

        if mask is not None:
            YX = YX * np.nan_to_num(mask_data[i].reshape(Y.shape[1]))

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

    output = np.empty((ncomp,) + data.shape[1:], np.float)

    for i in range(data.shape[1]):
        Y = data[:,i].reshape((data.shape[0], np.product(data.shape[2:])))
        U = np.dot(subVX, Y)

        if standardize:
            S2 = (project_resid(Y)**2).sum(0)
            Smhalf = np.nan_to_num(1./np.sqrt(S2)); del(S2)
            YX *= Smhalf

        if mask is not None:
            YX *= np.nan_to_num(mask_data[i].reshape(Y.shape[1]))

        U.shape = (U.shape[0],) + data.shape[2:]
        output[:,i] = U

    img_first_axis = image.axes.coord_names[0]
    # Rename the axis.

    # Because we started with XYZImage, all non-spatial
    # coordinates agree in the range and the domain
    # so this will work and the renamed_range
    # call is not even necessary because when we call
    # XYZImage, we only use the axisnames

    output_coordmap = image.coordmap.renamed_domain({img_first_axis:'PCA components'}).renamed_range({img_first_axis:'PCA components'})

    output_img = Image(output, output_coordmap)

    # We have to roll the axis back

    roll_index = xyz_image.axes.index(img_first_axis)
    output_img = image_rollaxis(output_img, roll_index, inverse=True)

    output_xyz = XYZImage(output_img.get_data(), 
                          xyz_image.affine,
                          output_img.axes.coord_names)

    return {'components over %s' % img_first_axis:time_series[:ncomp,],
            'pcnt_var': pcntvar,
            'images':output_xyz, 
            'rank':rank}






