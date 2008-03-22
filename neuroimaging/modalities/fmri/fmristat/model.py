"""
This module defines the two default
GLM passes of fmristat
"""

import numpy as np
from neuroimaging.fixes.scipy.stats_models.regression import OLSModel, ARModel
from neuroimaging.core.api import Image, data_generator, parcels

from neuroimaging.modalities.fmri.api import FmriImage, fmri_generator
from neuroimaging.core.api import f_generator
from neuroimaging.modalities.fmri.fmristat.delay import DelayContrast, \
     DelayContrastOutput
import neuroimaging.algorithms.statistics.regression as regression

from neuroimaging.algorithms.fwhm import fastFWHM
from neuroimaging.algorithms.statistics.regression import generate_ou


def model_generator(model, iterable=None):
    """
    Generator for the models for a pass of fmristat analysis.
    """
    for i, d in fmri_generator(model.data, iterable=iterable):
        if slicetimes is not None:
            rmodel = model.type(model.formula.design(frametimes + slicetimes[i]))
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

    def __init__(self, fmri_image, formula, resid=False, rho=True):
        self.type = OLSModel
        self.data = np.asarray(fmri_image) # this has some annoying overhead
        self.formula = formula
        self.outputs = []
        if resid:
            self.outputs.append(output_resid(blah))

    def execute(self):
        m = model_generator(self)
        r = results_generator(m)
        o = regression.generate_output(self.outputs, r)


def OLS_generator(fmri_image, formula):
    d = np.asarray(fmri_image) 

