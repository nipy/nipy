# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Misc tools to find activations and cut on maps
"""
from __future__ import absolute_import

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

import warnings

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
from scipy import stats, ndimage

# Local imports
from ..mask import largest_cc
from ..datasets.transforms.affine_utils import get_bounds


################################################################################
# Functions for automatic choice of cuts coordinates
################################################################################

def coord_transform(x, y, z, affine):
    """ Convert x, y, z coordinates from one image space to another space.

    Warning: x, y and z have Talairach ordering, not 3D numpy image ordering.

    Parameters
    ----------
    x : number or ndarray
        The x coordinates in the input space
    y : number or ndarray
        The y coordinates in the input space
    z : number or ndarray
        The z coordinates in the input space
    affine : 2D 4x4 ndarray
        affine that maps from input to output space.

    Returns
    -------
    x : number or ndarray
        The x coordinates in the output space
    y : number or ndarray
        The y coordinates in the output space
    z : number or ndarray
        The z coordinates in the output space
    """
    coords = np.c_[np.atleast_1d(x).flat, 
                   np.atleast_1d(y).flat, 
                   np.atleast_1d(z).flat,
                   np.ones_like(np.atleast_1d(z).flat)].T
    x, y, z, _ = np.dot(affine, coords)
    return x.squeeze(), y.squeeze(), z.squeeze()


def find_cut_coords(map, mask=None, activation_threshold=None):
    """ Find the center of the largest activation connect component.

        Parameters
        -----------
        map : 3D ndarray
            The activation map, as a 3D image.
        mask : 3D ndarray, boolean, optional
            An optional brain mask.
        activation_threshold : float, optional
            The lower threshold to the positive activation. If None, the 
            activation threshold is computed using find_activation.

        Returns
        -------
        x: float
            the x coordinate in voxels.
        y: float
            the y coordinate in voxels.
        z: float
            the z coordinate in voxels.
    """
    # To speed up computations, we work with partial views of the array,
    # and keep track of the offset
    offset = np.zeros(3) 
    # Deal with masked arrays:
    if hasattr(map, 'mask'):
        not_mask = np.logical_not(map.mask)
        if mask is None:
            mask = not_mask
        else:
            mask *= not_mask
        map = np.asarray(map)
    my_map = map.copy()
    if mask is not None:
        slice_x, slice_y, slice_z = ndimage.find_objects(mask)[0]
        my_map = my_map[slice_x, slice_y, slice_z]
        mask = mask[slice_x, slice_y, slice_z]
        my_map *= mask
        offset += [slice_x.start, slice_y.start, slice_z.start]
    # Testing min and max is faster than np.all(my_map == 0)
    if (my_map.max() == 0) and (my_map.min() == 0):
        return .5*np.array(map.shape)
    if activation_threshold is None:
        activation_threshold = stats.scoreatpercentile(
                                    np.abs(my_map[my_map !=0]).ravel(), 80)
    mask = np.abs(my_map) > activation_threshold-1.e-15
    mask = largest_cc(mask)
    slice_x, slice_y, slice_z = ndimage.find_objects(mask)[0]
    my_map = my_map[slice_x, slice_y, slice_z]
    mask = mask[slice_x, slice_y, slice_z]
    my_map *= mask
    offset += [slice_x.start, slice_y.start, slice_z.start]
    # For the second threshold, we use a mean, as it is much faster,
    # althought it is less robust
    second_threshold = np.abs(np.mean(my_map[mask]))
    second_mask = (np.abs(my_map)>second_threshold)
    if second_mask.sum() > 50:
        my_map *= largest_cc(second_mask)
    cut_coords = ndimage.center_of_mass(np.abs(my_map))
    return cut_coords + offset


################################################################################

def get_mask_bounds(mask, affine):
    """ Return the world-space bounds occupied by a mask given an affine.

        Notes
        -----

        The mask should have only one connect component.

        The affine should be diagonal or diagonal-permuted.
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(mask.shape, affine)
    slices = ndimage.find_objects(mask)
    if len(slices) == 0:
        warnings.warn("empty mask", stacklevel=2)
    else:
        x_slice, y_slice, z_slice = slices[0]
        x_width, y_width, z_width = mask.shape
        xmin, xmax = (xmin + x_slice.start*(xmax - xmin)/x_width,
                    xmin + x_slice.stop *(xmax - xmin)/x_width)
        ymin, ymax = (ymin + y_slice.start*(ymax - ymin)/y_width,
                    ymin + y_slice.stop *(ymax - ymin)/y_width)
        zmin, zmax = (zmin + z_slice.start*(zmax - zmin)/z_width,
                    zmin + z_slice.stop *(zmax - zmin)/z_width)

    return xmin, xmax, ymin, ymax, zmin, zmax


def _maximally_separated_subset(x, k):
    """
    Given a set of n points x = {x_1, x_2, ..., x_n} and a positive integer
    k < n, this function returns a subset of k points which are maximally
    spaced.

    Returns
    -------
    msssk: 1D array of k floats
        computed maximally-separated subset of k elements from x

    """

    # base cases
    if k < 1: raise ValueError("k = %i < 1 is senseless." % k)
    if k == 1: return [x[len(x) // 2]]

    # would-be maximally separated subset of k (not showing the terminal nodes)
    msss = list(range(1, len(x) - 1))

    # sorting is necessary for the heuristic to work
    x = np.sort(x)

    # iteratively delete points x_j of msss, for which x_(j + 1) - x_(j - 1) is
    # smallest, untill only k - 2 points survive
    while len(msss) + 2 > k:
        # survivors
        y = np.array([x[0]] + list(x[msss]) + [x[-1]])

        # remove most troublesome point
        msss = np.delete(msss, np.argmin(y[2:] - y[:-2]))

    # return maximally separated subset of k elements
    return x[[0] + list(msss) + [len(x) - 1]]


def find_maxsep_cut_coords(map3d, affine, slicer='z', n_cuts=None,
                           threshold=None):
    """ Heuristic finds `n_cuts` with max separation along a given axis

    Parameters
    ----------
    map3d : 3D array
        the data under consideration
    affine : array shape (4, 4)
        Affine mapping between array coordinates of `map3d` and real-world
        coordinates.
    slicer : string, optional
        sectional slicer; possible values are "x", "y", or "z"
    n_cuts : None or int >= 1, optional
        Number of cuts in the plot; if None, then a default value of 5 is
        forced.
    threshold : None or float, optional
        Thresholding to be applied to the map.  Values less than `threshold`
        set to 0.  If None, no thresholding applied.

    Returns
    -------
    cuts : 1D array of length `n_cuts`
        the computed cuts

    Raises
    ------
    ValueError:
        If `slicer` not in 'xyz'
    ValueError
        If `ncuts` < 1
    """
    if n_cuts is None:
        n_cuts = 5
    if n_cuts < 1:
        raise ValueError("n_cuts = %i < 1 is senseless." % n_cuts)
    # sanitize slicer
    if slicer not in 'xyz':
        raise ValueError(
            "slicer must be one of 'x', 'y', and 'z', got '%s'." % slicer)
    slicer = "xyz".index(slicer)

    # load data
    if map3d.ndim != 3:
        raise TypeError(
            "map3d must be 3D array, got shape %iD" % map3d.ndim)

    _map3d = np.rollaxis(map3d.copy(), slicer, start=3)
    _map3d = np.abs(_map3d)
    if threshold is not None:
        _map3d[_map3d < threshold] = 0

    # count activated voxels per plane
    n_activated_voxels_per_plane = np.array([(_map3d[..., z] > 0).sum()
                                    for z in range(_map3d.shape[-1])])
    perm = np.argsort(n_activated_voxels_per_plane)
    n_activated_voxels_per_plane = n_activated_voxels_per_plane[perm]
    good_planes = np.nonzero(n_activated_voxels_per_plane > 0)[0]
    good_planes = perm[::-1][:n_cuts * 4 if n_cuts > 1 else 1]

    # cast into coord space
    good_planes = np.array([
            # map cut coord into native space
            np.dot(affine,
                   np.array([0, 0, 0, 1]  # origin
                            ) + coord * np.eye(4)[slicer])[slicer]
            for coord in good_planes])

    # compute cut_coords maximally-separated planes
    return _maximally_separated_subset(good_planes, n_cuts)
