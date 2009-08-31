from routines import _texture, _histogram, texture_measures
from utils import clamp

###def _texture(ndarray im, ndarray H, size): 

import numpy as np


def texture(data, size=[3,3,3], method='drange', mask=None, th=0, bins=256):
    data_clamped, bins = clamp(data, th=th, mask=mask, bins=bins)
    hist = np.zeros(bins)
    _histogram(hist, data_clamped.flat)
    if method in texture_measures: 
        return _texture(data_clamped, hist, size, texture_measures[method])
    else: 
        return _texture(data_clamped, hist, size, texture_measures['custom'], method)




    
        
