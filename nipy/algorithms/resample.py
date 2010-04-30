"""
Some simple examples and utility functions for resampling.
"""

import copy

from scipy.ndimage import affine_transform
import numpy as np

from nipy.algorithms.interpolation import ImageInterpolator
from nipy.core.api import Image, CoordinateMap, AffineTransform, ArrayCoordMap, compose
import nipy.core.transforms.affines as affines


def resample_img2img(source, target, order=3):
    """ 
    This wraps the resample function to resample one image onto another
    the output of the function will give an image with shape of the target and data from the source
    
    Parameters
    ----------
    source : ``Image``
       Image instance that is to be resampled
    target : ``Image``
       Image instance to which source is resampled
       The output image will have the same shape as the target, and the same coordmap
    order : ``int``, optional
       What order of interpolation to use in `scipy.ndimage`

    Returns
    -------
    output : ``Image`` 
       Image with interpolated data and output.coordmap == target.coordmap 

    Examples
    --------      
    >>> from nipy.testing import funcfile, anatfile
    >>> from nipy.io.api import load_image
    >>> aimg_source = load_image(anatfile)
    >>> aimg_target = aimg_source
    >>> # in this case, we resample aimg to itself
    >>> resimg = resample_img2img(aimg_source, aimg_target)
    >>> 
    """
    sip, sop = source.coordmap.ndim
    tip, top = target.coordmap.ndim
    #print sip, sop, tip, top
    if sop != top:
        raise ValueError("source coordmap output dimension not equal to target coordmap output dimension")
    mapping = np.eye(sop+1) # this would usually be 3+1
    resimg = resample(source, target.coordmap, mapping, target.shape, order=order)
    return resimg


def resample(image, target, mapping, shape, order=3):
    """
    Resample an image to a target CoordinateMap with a "world-to-world" mapping
    and spline interpolation of a given order.

    Here, "world-to-world" refers to the fact that mapping should be
    a callable that takes a physical coordinate in "target"
    and gives a physical coordinate in "image". 

    Parameters
    ----------
    image : Image instance that is to be resampled
    target :target CoordinateMap for output image
    mapping : transformation from target.function_range
               to image.coordmap.function_range, i.e. 'world-to-world mapping'
               Can be specified in three ways: a callable, a
               tuple (A, b) representing the mapping y=dot(A,x)+b
               or a representation of this in homogeneous coordinates. 
    shape : shape of output array, in target.function_domain
    order : what order of interpolation to use in `scipy.ndimage`

    Returns
    -------
    output : Image instance with interpolated data and output.coordmap == target
                  
    """

    if not callable(mapping):
        if type(mapping) is type(()):
            A, b = mapping
            ndimout = b.shape[0]
            ndimin = A.shape[1]
            mapping  = np.zeros((ndimout+1, ndimin+1))
            mapping[:ndimout,:ndimin] = A
            mapping[:ndimout,-1] = b
            mapping[-1,-1] = 1.

     # image world to target world mapping

        TW2IW = AffineTransform(target.function_range, image.coordmap.function_range, mapping)
    else:
        if isinstance(mapping, AffineTransform):
            TW2IW = mapping
        else:
            TW2IW = CoordinateMap(target.function_range, image.coordmap.function_range, mapping)

    function_domain = target.function_domain
    function_range = image.coordmap.function_range

    # target voxel to image world mapping
    TV2IW = compose(TW2IW, target)

    # CoordinateMap describing mapping from target voxel to
    # image world coordinates

    if not isinstance(TV2IW, AffineTransform):
        # interpolator evaluates image at values image.coordmap.function_range,
        # i.e. physical coordinates rather than voxel coordinates

        grid = ArrayCoordMap.from_shape(TV2IW, shape)
        interp = ImageInterpolator(image, order=order)
        idata = interp.evaluate(grid.transposed_values)

        del(interp)
    else:
        TV2IV = compose(image.coordmap.inverse(), TV2IW)
        if isinstance(TV2IV, AffineTransform):
            A, b = affines.to_matrix_vector(TV2IV.affine)
            idata = affine_transform(np.asarray(image), A,
                                     offset=b,
                                     output_shape=shape,
                                     order=order)
        else:
            interp = ImageInterpolator(image, order=order)
            grid = ArrayCoordMap.from_shape(TV2IV, shape)
            idata = interp.evaluate(grid.values)
            del(interp)
            

    return Image(idata, copy.copy(target))

        
