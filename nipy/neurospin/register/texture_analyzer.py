from routines import _histogram
from utils import clamp, block3d, smaller_block3d

import numpy as np



class TextureAnalyzer: 

    def __init__(self, data, mask=None, th=0, bins=256): 
        
        self.data = data
        self.data_clamped, bins = clamp(data, th=th, mask=mask, bins=bins)
        self.hist = np.zeros(bins)
        # self.glcm = np.zeros([bins,bins])
        
    def eval(self): 
        _histogram(self.hist, self.data_clamped.flat)
        
