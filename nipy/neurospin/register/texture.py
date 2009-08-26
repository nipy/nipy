from routines import _texture, _histogram
from utils import clamp

###def _texture(ndarray im, ndarray H, size): 

import numpy as np


def texture(data, size=[3,3,3], method='entropy', mask=None, th=0, bins=256):
    data_clamped, bins = clamp(data, th=th, mask=mask, bins=bins)
    hist = np.zeros(bins)
    _histogram(hist, data_clamped.flat)
    return _texture(data_clamped, hist, size)
    
        
