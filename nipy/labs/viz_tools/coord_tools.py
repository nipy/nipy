# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Misc tools to find activations and cut on maps
"""

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
    """ Convert the x, y, z coordinates from one image space to another
        space. 
        
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

        Warning: The x, y and z have their Talairach ordering, not 3D
        numy image ordering.
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


def find_tubo_cut_coords(map3d, affine, slicer='z',
                         max_cuts=12, cut_sampling=1,
                         threshold=None):
    """
    Heuristic function to find cut_coords for which the total activation
    contained in the corresponding plane is optimal.

    map3d: 3D array
        the data under consideration

    slicer: string, optional (default "z")
        sectional slicer; possible values are "x", "y", or "z"

    max_cuts: int, optional (default 12)
        number of cuts in the plot

    cut_sampling: int, optional (default 1)
        spacing between cuts

    threshold: float, optional (default None)
        thresholding to be applied to the map

    Returns
    -------
    cut_coords: 1D array of length n_cuts
        the computed cut_coords

    Raises
    ------
    AssertionError

    """

    # sanitize slicer
    assert slicer in ['x', 'y', 'z'], "slice must be one of 'x', 'y', and 'z'"
    slicer = "xyz".index(slicer)

    # load data
    assert map3d.ndim == 3

    # slicer becomes last axis
    perm = np.arange(3)
    perm[-1], perm[slicer] = perm[slicer], perm[-1]
    map3d = np.transpose(map3d, axes=perm)

    # integrate over activated voxels in each slice
    total_activation_per_plane = [(map3d[..., ..., z] > threshold).sum()
                                  for z in xrange(map3d.shape[-1])]

    # order slices in descending order of the total activation contained
    # in the corresponding plane
    ordered_slices = np.argsort(total_activation_per_plane)[::-1]
    ordered_slices = np.array([

            # map cut coord into native space
            np.dot(affine,
                   np.array([0, 0, 0, 1]  # origin
                            ) + coord * np.eye(4)[slicer]
                   )[slicer]

            for coord in ordered_slices])

    return np.array(ordered_slices[:max_cuts:cut_sampling])
