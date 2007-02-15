''' Utilities for working with SPM '''

import numpy as N

from mlabwrap import mlab

from neuroimaging.core.api import Image

mlab.spm_defaults()

def image_to_vol(image):
    ''' Returns SPM vol struct version of image

    Accepts image as input
    Returns SPM vol struct as output
    '''
    image = Image(image)
    image_path = image._source.filename
    return mlab.spm_vol(image_path)
