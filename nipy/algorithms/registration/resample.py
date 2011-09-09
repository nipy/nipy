# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from scipy.ndimage import affine_transform, map_coordinates
from ...core.image.affine_image import AffineImage
from .image_utils import get_affine
from .affine import inverse_affine, Affine
from ._registration import (_cspline_transform,
                            _cspline_sample3d,
                            _cspline_resample3d)


INTERP_ORDER = 3


def resample(moving, transform, reference=None,
             mov_voxel_coords=False, ref_voxel_coords=False,
             dtype=None, interp_order=INTERP_ORDER):
    """
    Apply a transformation to the image considered as 'moving' to
    bring it into the same grid as a given `reference` image.

    This function uses scipy.ndimage except for the case
    `interp_order==3`, where a fast cubic spline implementation is
    used.

    Parameters
    ----------
    moving: nipy-like image
      Image to be resampled.
    transform: transform object
      Represents a transform that goes from the `reference` image to
      the `moving` image. It should have either an `apply` method, or
      an `as_affine` method.
    mov_voxel_coords : boolean
      True if the transform maps to voxel coordinates, False if it
      maps to world coordinates.
    ref_voxel_coords : boolean
      True if the transform maps from voxel coordinates, False if it
      maps from world coordinates.
    reference: nipy-like image
      Reference image, defaults to input.
    interp_order: number
      Spline interpolation order, defaults to 3.
    """
    if reference == None:
        reference = moving
    data = moving.get_data()
    if dtype == None:
        dtype = data.dtype

    # Case: affine transform
    if hasattr(transform, 'as_affine'):
        Tv = transform.as_affine()
        if not ref_voxel_coords:
            Tv = np.dot(Tv, get_affine(reference))
        if not mov_voxel_coords:
            Tv = np.dot(inverse_affine(get_affine(moving)), Tv)
        if interp_order == 3:
            output = _cspline_resample3d(data, reference.shape,
                                         Tv, dtype=dtype)
            output = output.astype(dtype)
        else:
            output = np.zeros(reference.shape, dtype=dtype)
            affine_transform(data, Tv[0:3, 0:3], offset=Tv[0:3, 3],
                             order=interp_order, cval=0,
                             output_shape=reference.shape, output=output)

    # Case: non-affine transform
    else:
        Tv = transform
        if not ref_voxel_coords:
            Tv = Tv.compose(Affine(get_affine(reference)))
        if not mov_voxel_coords:
            Tv = Affine(inverse_affine(get_affine(moving))).compose(Tv)
        coords = np.indices(reference.shape).transpose((1, 2, 3, 0))
        coords = np.reshape(coords, (np.prod(reference.shape), 3))
        coords = Tv.apply(coords).T
        if interp_order == 3:
            cbspline = _cspline_transform(data)
            output = np.zeros(reference.shape, dtype='double')
            output = _cspline_sample3d(output, cbspline, *coords)
            output = output.astype(dtype)
        else:
            output = map_coordinates(data, coords, order=interp_order,
                                     cval=0, output=dtype)

    return AffineImage(output, get_affine(reference), 'scanner')
