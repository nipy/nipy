# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from ..utils.affines import apply_affine
from .affine import Affine
from .grid_transform import GridTransform
from .affine import inverse_affine
from ._cubic_spline import cspline_transform, cspline_sample3d, cspline_resample3d
from .chain_transform import ChainTransform 
import numpy as np 
from scipy.ndimage import affine_transform, map_coordinates
from ...core.image.affine_image import AffineImage


INTERP_ORDER = 3
                   
def resample(moving, transform, grid_coords=False, reference=None, 
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
    
    grid_coords : boolean
      True if the transform maps grid coordinates, False if it maps
      world coordinates. 
    
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
        if not grid_coords: 
            Tv = np.dot(inverse_affine(moving.affine), np.dot(Tv, reference.affine))
        if interp_order == 3: 
            output = cspline_resample3d(data, reference.shape, Tv, dtype=dtype)
            output = output.astype(dtype)
        else: 
            output = np.zeros(reference.shape, dtype=dtype)
            affine_transform(data, Tv[0:3,0:3], offset=Tv[0:3,3],
                             order=interp_order, cval=0, 
                             output_shape=reference.shape, output=output)
    # Case: non-affine transform
    else:
        Tv = transform 
        if not grid_coords:
            Tv = Affine(inverse_affine(moving.affine)).compose(Tv.compose(reference.affine))
        coords = Tv.apply(np.indices(reference.shape).transpose((1,2,3,0)))
        coords = np.rollaxis(coords, 3, 0)
        if interp_order == 3: 
            cbspline = cspline_transform(data)
            output = np.zeros(reference.shape, dtype='double')
            output = cspline_sample3d(output, cbspline, *coords)
            output = output.astype(dtype)
        else: 
            output = map_coordinates(data, coords, order=interp_order, 
                                     cval=0, output=dtype)
    
    return AffineImage(output, reference.affine, 'scanner')


