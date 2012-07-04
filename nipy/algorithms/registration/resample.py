# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from scipy.ndimage import affine_transform, map_coordinates

from ...core.image.image_spaces import (make_xyz_image,
                                        as_xyz_image,
                                        xyz_affine)
from .affine import inverse_affine, Affine
from ._registration import (_cspline_transform,
                            _cspline_sample3d,
                            _cspline_resample3d)


INTERP_ORDER = 3


def resample(moving, transform=None, reference=None,
             mov_voxel_coords=False, ref_voxel_coords=False,
             dtype=None, interp_order=INTERP_ORDER):
    """ Resample `movimg` into voxel space of `reference` using `transform`

    Apply a transformation to the image considered as 'moving' to
    bring it into the same grid as a given `reference` image.  The
    transformation usually maps world space in `reference` to world space in
    `movimg`, but can also be a voxel to voxel mapping (see parameters below).

    This function uses scipy.ndimage except for the case `interp_order==3`,
    where a fast cubic spline implementation is used.

    Parameters
    ----------
    moving: nipy-like image
      Image to be resampled.
    transform: transform object or None
      Represents a transform that goes from the `reference` image to
      the `moving` image. None means an identity transform. Otherwise,
      it should have either an `apply` method, or an `as_affine`
      method. By default, `transform` maps between the output (world)
      space of `reference` and the output (world) space of `moving`.
      If `mov_voxel_coords` is True, maps to the *voxel* space of
      `moving` and if `ref_vox_coords` is True, maps from the *voxel*
      space of `reference`.
    reference : None or nipy-like image or tuple, optional
      The reference image defines the image dimensions and xyz affine
      to which to resample. It can be input as a nipy-like image or as
      a tuple (shape, affine). If None, use `movimg` to define these.
    mov_voxel_coords : boolean, optional
      True if the transform maps to voxel coordinates, False if it
      maps to world coordinates.
    ref_voxel_coords : boolean, optional
      True if the transform maps from voxel coordinates, False if it
      maps from world coordinates.
    interp_order: int, optional
      Spline interpolation order, defaults to 3.

    Returns
    -------
    aligned_img : Image
        Image resliced to `reference` with reference-to-movimg transform
        `transform`
    """
    # Function assumes xyz_affine for inputs
    moving = as_xyz_image(moving)
    mov_aff = xyz_affine(moving)
    if reference == None:
        reference = moving
    if isinstance(reference, (tuple, list)):
        ref_shape, ref_aff = reference
    else:
        # Expecting image. Must be an image that can make an xyz_affine
        reference = as_xyz_image(reference)
        ref_shape = reference.shape
        ref_aff = xyz_affine(reference)
    if not len(ref_shape) == 3 or not ref_aff.shape == (4, 4):
        raise ValueError('Input image should be 3D')
    data = moving.get_data()
    if dtype == None:
        dtype = data.dtype

    # Assume identity transform by default
    if transform == None:
        transform = Affine()

    # Case: affine transform
    if hasattr(transform, 'as_affine'):
        Tv = transform.as_affine()
        if not ref_voxel_coords:
            Tv = np.dot(Tv, ref_aff)
        if not mov_voxel_coords:
            Tv = np.dot(inverse_affine(mov_aff), Tv)
        if interp_order == 3:
            output = _cspline_resample3d(data, ref_shape,
                                         Tv, dtype=dtype)
            output = output.astype(dtype)
        else:
            output = np.zeros(ref_shape, dtype=dtype)
            affine_transform(data, Tv[0:3, 0:3], offset=Tv[0:3, 3],
                             order=interp_order, cval=0,
                             output_shape=ref_shape, output=output)

    # Case: non-affine transform
    else:
        Tv = transform
        if not ref_voxel_coords:
            Tv = Tv.compose(Affine(ref_aff))
        if not mov_voxel_coords:
            Tv = Affine(inverse_affine(mov_aff)).compose(Tv)
        coords = np.indices(ref_shape).transpose((1, 2, 3, 0))
        coords = np.reshape(coords, (np.prod(ref_shape), 3))
        coords = Tv.apply(coords).T
        if interp_order == 3:
            cbspline = _cspline_transform(data)
            output = np.zeros(ref_shape, dtype='double')
            output = _cspline_sample3d(output, cbspline, *coords)
            output = output.astype(dtype)
        else:
            output = map_coordinates(data, coords, order=interp_order,
                                     cval=0, output=dtype)

    return make_xyz_image(output, ref_aff, 'scanner')
