# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Some simple examples and utility functions for resampling.
"""

import copy

import numpy as np

from scipy.ndimage import affine_transform

from .interpolation import ImageInterpolator
from ..core.api import (Image, CoordinateMap, AffineTransform,
                        ArrayCoordMap, compose)
from ..core.transforms import affines

def resample_img2img(source, target, order=3):
    """  Resample `source` image to space of `target` image

    This wraps the resample function to resample one image onto another.
    The output of the function will give an image with shape of the
    target and data from the source

    Parameters
    ----------
    source : ``Image``
       Image instance that is to be resampled
    target : ``Image``
       Image instance to which source is resampled The output image will
       have the same shape as the target, and the same coordmap
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
    """
    sip, sop = source.coordmap.ndims
    tip, top = target.coordmap.ndims
    #print sip, sop, tip, top
    if sop != top:
        raise ValueError("source coordmap output dimension not equal "
                         "to target coordmap output dimension")
    mapping = np.eye(sop+1) # this would usually be 3+1
    resimg = resample(source, target.coordmap, mapping, target.shape, order=order)
    return resimg


def resample(image, target, mapping, shape, order=3):
    """ Resample `image` to `target` CoordinateMap

    Use a "world-to-world" mapping `mapping` and spline interpolation of a 
    `order`.

    Here, "world-to-world" refers to the fact that mapping should be a
    callable that takes a physical coordinate in "target" and gives a
    physical coordinate in "image".

    Parameters
    ----------
    image : Image instance
       image that is to be resampled
    target : CoordinateMap
       coordinate map for output image
    mapping : callable or tuple or array
       transformation from target.function_range to
       image.coordmap.function_range, i.e. 'world-to-world mapping'. Can
       be specified in three ways: a callable, a tuple (A, b)
       representing the mapping y=dot(A,x)+b or a representation of this
       mapping as an affine array, in homogeneous coordinates.
    shape : sequence of int
       shape of output array, in target.function_domain
    order : int, optional
       what order of interpolation to use in `scipy.ndimage`

    Returns
    -------
    output : Image instance
       with interpolated data and output.coordmap == target
    """
    if not callable(mapping):
        if type(mapping) is type(()):
            mapping = affines.from_matrix_vector(*mapping)
        # image world to target world mapping
        TW2IW = AffineTransform(target.function_range,
                                image.coordmap.function_range,
                                mapping)
    else:
        if isinstance(mapping, AffineTransform):
            TW2IW = mapping
        else:
            TW2IW = CoordinateMap(target.function_range,
                                  image.coordmap.function_range,
                                  mapping)
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
    else: # it is an affine transform, but, what if we compose?
        TV2IV = compose(image.coordmap.inverse(), TV2IW)
        if isinstance(TV2IV, AffineTransform): # still affine
            A, b = affines.to_matrix_vector(TV2IV.affine)
            idata = affine_transform(np.asarray(image), A,
                                     offset=b,
                                     output_shape=shape,
                                     order=order)
        else: # not affine anymore
            interp = ImageInterpolator(image, order=order)
            grid = ArrayCoordMap.from_shape(TV2IV, shape)
            idata = interp.evaluate(grid.values)
            del(interp)
    return Image(idata, copy.copy(target))
