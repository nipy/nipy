from routines import _texture, _histogram
from utils import clamp

###def _texture(ndarray im, ndarray H, size): 

import numpy as np



class TextureAnalyzer: 

    def __init__(self, data, mask=None, th=0, bins=256): 
        
        self.data = data
        self.data_clamped, bins = clamp(data, th=th, mask=mask, bins=bins)
        self.hist = np.zeros(bins)
        # self.glcm = np.zeros([bins,bins])
        self.set_size()

    def histogram(self): 
        _histogram(self.hist, self.data_clamped.flat)
        
    def eval(self): 
        return _texture(self.data_clamped, self.hist, self.size)

    def set_size(self, size=[3,3,3]): 
        self.size = size
        
