"""
This module defines the two default
GLM passes of fmristat
"""

import gc, copy

import numpy as np
from neuroimaging.fixes.scipy.stats.models.regression import OLSModel, ARModel
from neuroimaging.core.api import data_generator, parcels, matrix_generator

from neuroimaging.modalities.fmri.api import FmriImageList, fmri_generator
from neuroimaging.core.api import f_generator, fromarray, save_image, Image
from neuroimaging.core.reference.api import Affine, CoordinateMap
from neuroimaging.modalities.fmri.fmristat.delay import DelayContrast, \
     DelayContrastOutput
import neuroimaging.algorithms.statistics.regression as regression

from neuroimaging.algorithms.fwhm import fastFWHM
import neuroimaging.algorithms.statistics.regression as regression


# FIXME: Is there a clean, generic way to handle these temporary
# files?  We removed create_outfile from image.py because we'd like to
# minimize the number of ways users create and deal with files.
# Simplify the api.  But perhaps there is a need for a create_outfile
# type function?
def _create_outfile(filename, coordmap):
    img = fromarray(np.zeros(coordmap.shape), coordmap=coordmap)
    return save_image(img, filename)

def model_generator(formula, data, volume_start_times, iterable=None, slicetimes=None,
                    model_type=OLSModel, model_params = lambda x: ()):
    """
    Generator for the models for a pass of fmristat analysis.
    """
    for i, d in matrix_generator(fmri_generator(data, iterable=iterable)):
        model_args = model_params(i) # model may depend on i
        rmodel = model_type(formula.design(volume_start_times), *model_args)
        yield i, d, rmodel

def results_generator(model_iterable):
    """
    Generator for results from an iterator that returns
    (index, data, model) tuples.

    See model_generator.
    """
    for i, d, m in model_iterable:
        yield i, m.fit(d)

class OLS:
    """
    First pass through fmri_image.

    Parameters
    ----------
    fmri_image : `FmriImageList`
    formula :  `neuroimaging.modalities.fmri.protocol.Formula`


    """
    def __init__(self, fmri_image, formula, outputs=[]):
        self.fmri_image = fmri_image
        self.data = np.asarray(fmri_image)
        self.formula = formula
        self.outputs = outputs

    def execute(self):
        m = model_generator(self.formula, self.data,
                            self.fmri_image.volume_start_times,
                            model_type=OLSModel)
        r = results_generator(m)

        def reshape(i, x):
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

        o = generate_output(self.outputs, r, reshape=reshape)

class AR1:

    """
    Second pass through fmri_image.

    Parameters
    ----------
    fmri_image : `FmriImageList`
    formula :  `neuroimaging.modalities.fmri.protocol.Formula`
    rho : Image of AR(1) coefficients.
    """

    def __init__(self, fmri_image, formula, rho, outputs=[]):
        self.fmri_image = fmri_image
        self.data = np.asarray(fmri_image)
        self.formula = formula
        self.outputs = outputs
        self.rho = rho

    def execute(self):
        iterable = parcels((np.asarray(self.rho)*100).astype(np.int) / 100.)
        def model_params(i):
            return (np.asarray(self.rho)[i].mean(),)
        m = model_generator(self.formula, self.data,
                            self.fmri_image.volume_start_times,
                            iterable=iterable,
                            model_type=ARModel,
                            model_params=model_params)
        r = results_generator(m)

        o = generate_output(self.outputs, r, reshape=reshape)

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

def output_T(outbase, contrast, fmri_image, effect=True, sd=True, t=True,
             clobber=False):
    """
    outbase: a string interpolator object with key %(stat)s
    contrast: a TContrast
    """
    if effect:
        effectim = _create_outfile(outbase % {'stat':'effect'},
                                   fmri_image[0].coordmap)
    else:
        effectim = None

    if sd:
        sdim = _create_outfile(outbase % {'stat':'sd'}, fmri_image[0].coordmap)
    else:
        sdim = None

    if t:
        tim = _create_outfile(outbase % {'stat':'t'}, fmri_image[0].coordmap)
    else:
        tim = None
    return regression.TOutput(contrast, effect=effectim,
                              sd=sdim, t=tim)

def output_F(outfile, contrast, fmri_image, clobber=False):
    f = _create_outfile(outfile, fmri_image[0].coordmap)
    c = copy.deepcopy(contrast)
    return regression.RegressionOutput(f, lambda x: regression.output_F(x, c))
                             
def output_AR1(outfile, fmri_image, clobber=False):
    """
    Create an output file of the AR1 parameter from the OLS pass of
    fmristat.

    image: FmriImageList 

    """
    outim = _create_outfile(outfile, fmri_image[0].coordmap)
    return regression.RegressionOutput(outim, regression.output_AR1)

def output_resid(outfile, fmri_image, clobber=False):
    """
    Create an output file of the residuals parameter from the OLS pass of
    fmristat.

    Uses affine part of the first image to output resids unless
    fmri_image is an Image.

    """

    if isinstance(fmri_image, FmriImageList):
        n = len(fmri_image.list)
        T = np.zeros((5,5))
        g = fmri_image[0].coordmap
        T[1:,1:] = fmri_image[0].coordmap.affine
        T[0,0] = (fmri_image.volume_start_times[1:] - fmri_volume_start_times[:-1]).mean()
        anames = ["time"] + [a.name for a in g.input_coords.axes()]
        coordmap = CoordinateMap.from_affine(Affine(T),
                                        anames,
                                        (n,) + g.shape)
    elif isinstance(fmri_image, Image):
        coordmap = fmri_image.coordmap
    else:
        raise ValueError, "expecting FmriImageList or 4d Image"
    outim = _create_outfile(outfile, coordmap)
    return regression.ArrayOutput(outim, regression.output_resid)

def generate_output(outputs, iterable, reshape=lambda x, y: (x, y)):
    """
    Write out results of a given output.

    In the regression setting, results is generally
    going to be a scipy.stats.models.model.LikelihoodModelResults instance.
    """
    for i, results in iterable:
        for output in outputs:
            if not hasattr(output, "list"): # lame test here
                i, d = reshape(i, output(results))
                output[i] = d
            else:
                r = output(results)
                for j, l in enumerate(output.list):
                    i, d = reshape(i, r[j])
                    l[i] = d




