import numpy as np

from nipy.neurospin.register.iconic_matcher import IconicMatcher
from nipy.neurospin.register.routines import cspline_resample
from nipy.neurospin.register.realign4d import Image4d, realign4d, _resample4d

# Use the brifti image object
from nipy.io.imageformats import Nifti1Image as Image 


def image4d(im, tr, tr_slices=None, start=0.0, 
            slice_order='ascending', interleaved=False):

    """
    Wrapper function. 
    Returns an Image4d instance. 

    Assumes that the input image referential is 'scanner' and that the
    third array index stands for 'z', i.e. the slice index. 
    """
    return Image4d(im.get_data(), im.get_affine(),
                   tr=tr, tr_slices=tr_slices, start=start,
                   slice_order=slice_order, interleaved=interleaved)


def resample4d(im4d, transforms=None): 
    """
    corr_img = resample4d(im4d, transforms=None)
    """
    return Image(_resample4d(im4d, transforms),
                 im4d.get_affine())
