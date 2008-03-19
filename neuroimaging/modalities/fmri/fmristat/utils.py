__docformat__ = 'restructuredtext'

import gc, os, fpformat

import numpy as N
import numpy.linalg as L
from neuroimaging.fixes.scipy.stats_models.regression import OLSModel, ARModel
from neuroimaging.fixes.scipy.stats_models.utils import monotone_fn_inverter, rank 

from neuroimaging.modalities.fmri.api import FmriImage
from neuroimaging.modalities.fmri.fmristat.delay import DelayContrast, \
     DelayContrastOutput
from neuroimaging.algorithms.statistics.regression import LinearModelIterator
from neuroimaging.modalities.fmri.regression import AROutput, \
     TContrastOutput, FContrastOutput, ResidOutput
from neuroimaging.core.api import Image
from neuroimaging.algorithms.fwhm import fastFWHM


class WholeBrainNormalize(object):
    """
    TODO
    """
    
    def __init__(self, fmri_image, mask=None):
        """
        :Parameters:
            `fmri_image` : TODO
                TODO
            `mask` : TODO
                TODO
        """
        if mask is not None:
            mask = mask[:]
            nvox = mask.astype(N.int32).sum()
        else:
            nvox = N.product(fmri_image.grid.shape[1:])

        self.n = fmri_image.grid.shape[0]
        self.avg = N.zeros((self.n,))

        for i in range(self.n):
            d = fmri_image[i:i+1]
            if mask is not None:
                d = d * mask # can't do in place as the slice points into a 
                             # memmap which may not be writable.
            self.avg[i] = d.sum() / nvox

    def __call__(self, fmri_data):
        """
        :Parameters:
            `fmri_data` : TODO
                TODO
                
        :Returns: TODO
            TODO
        """
        out = N.zeros(fmri_data.shape)
        for i in range(self.n):
            out[i] = fmri_data[i] * 100. / self.avg[i]
        return out




