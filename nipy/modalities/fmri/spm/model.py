# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import gc, os, fpformat, time

import numpy as np
import numpy.linalg as L
from scipy.stats import f as FDbn
from nipy.algorithms.statistics.models.regression import OLSModel, GLSModel

from nipy.core.api import Image
from nipy.modalities.fmri.fmristat import model as fmristat
from nipy.modalities.fmri.fmristat.model import OLS
import correlation, reml


def Fmask(Fimg, dfnum, dfdenom, pvalue=1.0e-04):
    """
    Create mask for use in estimating pooled covariance based on
    an F contrast.
    """

    ## TODO check nipy.algorithms.statistics.models.contrast to see if rank is
    ## correctly set -- I don't think it is right now.
    print dfnum, dfdenom
    thresh = FDbn.ppf(pvalue, dfnum, dfdenom)
    return Image(np.greater(np.asarray(Fimg), thresh), Fimg.grid.copy())


def estimate_pooled_covariance(resid, ARtarget=[0.3], mask=None):
    """
    Use SPM's REML implementation to estimate a pooled covariance matrix.
    
    Thresholds an F statistic at a marginal pvalue to estimate
    covariance matrix.

    """
    resid 
    n = resid[:].shape[0]
    components = correlation.ARcomponents(ARtarget, n)

    raw_sigma = 0
    nvox = 0
    for i in range(resid.shape[1]):
        d = np.asarray(resid[:,i])
        d.shape = (d.shape[0], np.product(d.shape[1:]))
        keep = np.asarray(mask[i])
        keep.shape = np.product(keep.shape)
        d = d.compress(keep, axis=1)
        raw_sigma += np.dot(d, d.T)
        nvox += d.shape[1]
    raw_sigma /= nvox
    C, h, _ = reml.reml(raw_sigma,
                        components,
                        n=nvox)
    return C

class SecondStage(object):
    """
    Parameters
    ----------
    fmri_image : `FmriImageList`
       object returning 4D array from ``np.asarray``, having attribute
       ``volume_start_times`` (if `volume_start_times` is None), and
       such that ``object[0]`` returns something with attributes ``shape``
    formula :  :class:`nipy.algorithms.statistics.formula.Formula`
    sigma : 
    outputs :
    volume_start_times : 
    """

    def __init__(self, fmri_image, formula, sigma, outputs=[],
                 volume_start_times=None):
        self.fmri_image = fmri_image
        self.data = np.asarray(fmri_image)
        self.formula = formula
        self.outputs = outputs
        self.sigma = sigma

        if volume_start_times is None:
            self.volume_start_times = self.fmri_image.volume_start_times
        else:
            self.volume_start_times = volume_start_times

    def execute(self):
        def model_params(*args):
            return (self.sigma,)
        m = fmristat.model_generator(self.formula, self.data,
                                     self.volume_start_times,
                                     model_type=GLSModel,
                                     model_params=model_params)
        r = fmristat.results_generator(m)

        def reshape(i, x):
            """
            To write output, arrays have to be reshaped --
            this function does the appropriate reshaping for the two
            passes of fMRIstat.

            These passes are i) 'slices through the z-axis'
                            ii) 'parcels of approximately constant AR1 coefficient'
            """
    
            if len(x.shape) == 2:
                if type(i) is type(1):
                    x.shape = (x.shape[0],) + self.fmri_image[0].shape[1:]                        
                if type(i) not in [type([]), type(())]:
                    i = (i,)
                else:
                    i = tuple(i)
                i = (slice(None,None,None),) + tuple(i)
            else:
                if type(i) is type(1):
                    x.shape = self.fmri_image[0].shape[1:]
            return i, x

        o = fmristat.generate_output(self.outputs, r, reshape=reshape)


