# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from ..utils.affines import apply_affine
from .affine import Affine
from .grid_transform import GridTransform
from ._cubic_spline import cspline_transform, cspline_sample3d, cspline_resample3d

from ...core.image.affine_image import AffineImage

import numpy as np 
from scipy.ndimage import affine_transform, map_coordinates

_INTERP_ORDER = 3
                   
def resample(moving, transform, grid_coords=False, reference=None, 
             dtype=None, interp_order=_INTERP_ORDER):
    """
    Apply a transformation to the image considered as 'moving' to
    bring it into the same grid as a given 'reference' image. 

    This function uses scipy.ndimage except for the case
    `interp_order==3`, where a fast cubic spline implementation is
    used.
    
    Parameters
    ----------
    moving: nipy-like image
      Image to be resampled. 

    transform: nd array
      Either a 4x4 matrix describing an affine transformation,
      or an array with last dimension 3 describing voxelwise
      displacements of the reference grid points.
      For technical reasons, the transform is assumed to go from the
      'reference' to the 'moving'.
    
    grid_coords : boolean
      True if the transform maps to grid coordinates, False if it maps
      to world coordinates
    
    reference: nipy-like image 
      Reference image, defaults to input. 
      
    interp_order: number 
      Spline interpolation order, defaults to 3. 
    """
    if reference == None: 
        reference = moving
    shape = reference.shape
    data = moving.get_data()
    if dtype == None: 
        dtype = data.dtype
    if isinstance(transform, Affine): 
        affine = True
        t = transform.as_affine()
    elif isinstance(transform, GridTransform): 
        affine = False
        t = transform.as_displacements() 
    else: 
        t = np.asarray(transform)
        affine = t.shape[-1] == 4
    inv_affine = np.linalg.inv(moving.affine)

    # Case: affine transform
    if affine: 
        if not grid_coords:
            t = np.dot(inv_affine, np.dot(t, reference.affine))
        if interp_order == 3: 
            output = cspline_resample3d(data, shape, t, dtype=dtype)
            output = output.astype(dtype)
        else: 
            output = np.zeros(shape, dtype=dtype)
            affine_transform(data, t[0:3,0:3], offset=t[0:3,3],
                             order=interp_order, cval=0, 
                             output_shape=shape, output=output)
    
    # Case: precomputed displacements
    else:
        if not grid_coords:
            t = apply_affine(inv_affine, t)
        coords = np.rollaxis(t, 3, 0)
        if interp_order == 3: 
            cbspline = cspline_transform(data)
            output = np.zeros(shape, dtype='double')
            output = cspline_sample3d(output, cbspline, *coords)
            output = output.astype(dtype)
        else: 
            output = map_coordinates(data, coords, order=interp_order, 
                                     cval=0, output=dtype)
    
    return AffineImage(output, reference.affine, 'scanner')


