"""
Some simple examples and utility functions for resampling.
"""

from scipy.ndimage import affine_transform
import numpy as np

from neuroimaging.algorithms.interpolation import ImageInterpolator
from neuroimaging.core.api import Image, CoordinateMap, Mapping, Affine 

def resample(image, target, mapping, order=3):
    """
    Resample an image to a target CoordinateMap with a "world-to-world" mapping
    and spline interpolation of a given order.

    Here, "world-to-world" refers to the fact that mapping should be
    a callable that takes a physical coordinate in "target"
    and gives a physical coordinate in "image". 

    INPUTS:
    -------
    image -- Image instance that is to be resampled
    target -- target CoordinateMap for output image
    mapping -- transformation from target.output_coords
               to image.coordmap.output_coords, i.e. 'world-to-world mapping'
               Can be specified in three ways: a callable, a
               tuple (A, b) representing the mapping y=dot(A,x)+b
               or a representation of this in homogeneous coordinates. 
    order -- what order of interpolation to use in `scipy.ndimage`

    OUTPUTS:
    --------
    output -- Image instance with interpolated data and output.coordmap == target
                  
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
        mapping = Affine(mapping)

    input_coords = target.input_coords
    output_coords = image.coordmap.output_coords

    # image world to target world mapping
    
    TW2IW = Mapping.from_callable(mapping)

    # target voxel to image world mapping
    TV2IW = TW2IW * target.mapping

    # CoordinateMap describing mapping from target voxel to
    # image world coordinates

    output_coordmap = CoordinateMap(TV2IW,
                               target.input_coords,
                               image.coordmap.output_coords)

    if not isinstance(TV2IW, Affine):
        # interpolator evaluates image at values image.coordmap.output_coords,
        # i.e. physical coordinates rather than voxel coordinates

        interp = ImageInterpolator(image, order=order)
        idata = interp.evaluate(output_coordmap.range())
        del(interp)
    else:
        TV2IV = image.coordmap.mapping.inverse() * TV2IW
        if isinstance(TV2IV, Affine):
            A, b = TV2IV.params
            idata = affine_transform(np.asarray(image), A,
                                     offset=b,
                                     output_shape=output_coordmap.shape)
        else:
            interp = ImageInterpolator(image, order=order)
            idata = interp.evaluate(output_coordmap.range())
            del(interp)
            
    return Image(idata, target.copy())

        
