import gc, os, fpformat, time

import numpy as np
import numpy.linalg as L
from scipy.stats import f as FDbn
from neuroimaging.fixes.scipy.stats.models.regression import OLSModel, GLSModel

from neuroimaging.core.api import Image
from neuroimaging.modalities.fmri.fmristat import model as fmristat
from neuroimaging.modalities.fmri.fmristat.model import OLS
import correlation, reml

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


def Fmask(Fimg, dfnum, dfdenom, pvalue=1.0e-04):
    """
    Create mask for use in estimating pooled covariance based on
    an F contrast.
    """

    ## TODO check neuroimaging.fixes.scipy.stats.models.contrast to see if rank is
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

class SecondStage:

    """
    Second pass through fmri_image.

    Parameters
    ----------
    fmri_image : `FmriImage`
    formula :  `neuroimaging.modalities.fmri.protocol.Formula`
    rho : Image of AR(1) coefficients.
    """

    def __init__(self, fmri_image, formula, sigma, outputs=[],
                 frametimes=None):
        self.fmri_image = fmri_image
        self.data = np.asarray(fmri_image)
        self.formula = formula
        self.outputs = outputs
        self.sigma = sigma

        if frametimes is None:
            self.frametimes = self.fmri_image.frametimes
        else:
            self.frametimes = frametimes

    def execute(self):
        def model_params(*args):
            return (self.sigma,)
        m = fmristat.model_generator(self.formula, self.data,
                                     self.frametimes,
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
            
