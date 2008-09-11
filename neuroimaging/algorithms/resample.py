"""
Some simple examples and utility functions for resampling.
"""

from scipy.ndimage import affine_transform
import numpy as np

from neuroimaging.algorithms.interpolation import ImageInterpolator as I
from neuroimaging.core.api import Image, SamplingGrid, Mapping, Affine

def resample(image, target, mapping, order=3):
    """
    Resample an image to a target SamplingGrid with a "world-to-world" mapping
    and spline interpolation of a given order.

    Here, "world-to-world" refers to the fact that mapping should be
    a callable that takes a physical coordinate in "target"
    and gives a physical coordinate in "image". 

    INPUTS:
    -------
    image -- Image instance that is to be resampled
    target -- target SamplingGrid for output image
    mapping -- transformation from target.output_coords
               to image.grid.output_coords, i.e. 'world-to-world mapping'
    order -- what order of interpolation 

    OUTPUTS:
    --------
    output -- Image instance with interpolated data and output.grid == target
                  
    """

    input_coords = target.input_coords
    output_coords = image.grid.output_coords

    # image world to target world mapping
    
    TW2IW = mapping

    # target voxel to image world mapping
    
    if isinstance(mapping, Mapping):
        TV2IW = TW2IW * target.mapping
    else:
        TV2IW = lambda x: TW2IW(target.mapping(x))

    # SamplingGrid describing mapping from target voxel to
    # image world coordinates

    output_grid = SamplingGrid(TV2IW,
                               target.input_coords,
                               image.grid.output_coords)

    if not isinstance(TV2IW, Affine):
        # interpolator evaluates image at values image.grid.output_coords,
        # i.e. physical coordinates rather than voxel coordinates

        interp = I(image, order=order)
        idata = interp(output_grid.range())
        del(interp)
    else:
        TV2IV = image.grid.mapping.inverse() * TV2IW
        if isinstance(TV2IW, Affine):
            A, b = TV2IV.params
            idata = affine_transform(np.asarray(image), A,
                                     offset=b,
                                     output_shape=output_grid.shape)
        else:
            interp = I(image, order=order)
            idata = interp(output_grid.range())
            del(interp)
            
    return Image(idata, target.copy())

def affine(image, target, transform, order=3):
    """
    Perform affine resampling using splines of a given order,
    resampling image to a grid target based on transform.
    
    INPUTS
    ------
    image -- `Image` to be resampled
    target -- `SamplingGrid` instance for output
    transform -- either a tuple (A, b) representing the mapping y=dot(A,x)+b
                 or a representation of this in homogeneous coordinates, 
                 the transformation is from target.output_coords to
                 image.grid.input_coords
    """
    if type(transform) is type(()):
        A, b = transform
        ndim = b.shape[0]
        transform  = np.identity(ndim+1)
        transform[:ndim,:ndim] = A
        transform[:ndim,-1] = b
    mapping = Affine(transform)
    return resample(image, target, mapping, order=order)
        
