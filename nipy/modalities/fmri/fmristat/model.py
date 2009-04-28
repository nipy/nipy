"""
This module defines the two default
GLM passes of fmristat
"""

import os.path as path

import numpy as np
from scipy.linalg import toeplitz

from nipy.fixes.scipy.stats.models.regression import OLSModel, ARModel
from nipy.fixes.scipy.stats.models.utils import recipr

# nipy core imports

from nipy.core.api import Image, data_generator, parcels, matrix_generator
from nipy.core.api import f_generator, Image
from nipy.core.api import Affine, CoordinateMap

# nipy IO imports

from nipy.io.api import  save_image, load_image

# fmri imports

from nipy.modalities.fmri.api import FmriImageList, fmri_generator
from nipy.modalities.fmri.fmristat.delay import DelayContrast, \
     DelayContrastOutput

import nipy.algorithms.statistics.regression as regression
from nipy.algorithms.fwhm import fastFWHM
import nipy.algorithms.statistics.regression as regression


class ModelOutputImage:

    """
    These images have their values filled in as the model is fit, and
    are saved to disk after being completely filled in.

    They are saved to disk by calling the 'save' method.

    The __getitem__ and __setitem__ calls are delegated to a private Image.
    An exception is raised if trying to get/set data after the data has been saved to disk.
    
    """

    def __init__(self, filename, coordmap, shape, clobber=False):
        self.filename = filename
        self._im = Image(np.zeros(shape), coordmap)
        self.clobber = clobber
        self._flushed = False

    def save(self):
        """
        Save current Image data to disk as a .nii file.
        """

        if not self.clobber and path.exists(self.filename):
            raise ValueError, 'trying to clobber existing file'

        save_image(self._im, self.filename)

        self._flushed = True
        del(self._im)

    def __getitem__(self, item):
        if not self._flushed:
            return self._im[item]
        else:
            raise ValueError, 'trying to read value from a saved ModelOutputImage'

    def __setitem__(self, item, value):
        if not self._flushed:
            self._im[item] = value
        else:
            raise ValueError, 'trying to set value on saved ModelOutputImage'
        

def model_generator(formula, data, volume_start_times, iterable=None, 
                    slicetimes=None, model_type=OLSModel, 
                    model_params = lambda x: ()):
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
    formula :  `nipy.modalities.fmri.protocol.Formula`

    """

    def __init__(self, fmri_image, formula, outputs=[], 
                 volume_start_times=None):
        self.fmri_image = fmri_image
        self.data = np.asarray(fmri_image)
        self.formula = formula
        self.outputs = outputs

        if volume_start_times is None:
            self.volume_start_times = self.fmri_image.volume_start_times
        else:
            self.volume_start_times = volume_start_times
            
    def execute(self):
        m = model_generator(self.formula, self.data,
                            self.volume_start_times,
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


def estimateAR(resid, design, order=1):
    """
    Estimate AR parameters using bias correction from fMRIstat.

    Inputs:
    -------

    resid:  residual image
    model:  an OLS model used to estimate residuals
    """

    p = order

    R = np.identity(design.shape[0]) - np.dot(design, np.linalg.pinv(design))
    M = np.zeros((p+1,)*2)
    I = np.identity(R.shape[0])

    for i in range(p+1):
        Di = np.dot(R, toeplitz(I[i]))
        for j in range(p+1):
            Dj = np.dot(R, toeplitz(I[j]))
            M[i,j] = np.diagonal((np.dot(Di, Dj))/(1.+(i>0))).sum()
                    
    invM = np.linalg.inv(M)

    rresid = np.asarray(resid).reshape(resid.shape[0], 
                                       np.product(resid.shape[1:]))
    sum_sq = np.sum(rresid**2, axis=0)

    cov = np.zeros((p + 1,) + sum_sq.shape)
    cov[0] = sum_sq
    for i in range(1, p+1):
        cov[i] = np.add.reduce(rresid[i:] * rresid[0:-i], 0)
    cov = np.dot(invM, cov)
    output = cov[1:] * recipr(cov[0])
    output = np.squeeze(output)
    output.shape = resid.shape[1:]
    return output


class AR1:

    """
    Second pass through fmri_image.

    Parameters
    ----------
    fmri_image : `FmriImageList`
    formula :  `nipy.modalities.fmri.protocol.Formula`
    rho : Image of AR(1) coefficients.
    """

    def __init__(self, fmri_image, formula, rho, outputs=[],
                 volume_start_times=None):
        self.fmri_image = fmri_image


        self.data = np.asarray(fmri_image)

        self.formula = formula
        self.outputs = outputs

        # Cleanup rho values, truncate them to a scale of 0.01
        g = rho.coordmap.copy()
        rho = np.asarray(rho)
        m = np.isnan(rho)
        r = (np.clip(rho,-1,1) * 100).astype(np.int) / 100.
        r[m] = np.inf
        self.rho = Image(r, g)

        if volume_start_times is None:
            self.volume_start_times = self.fmri_image.volume_start_times
        else:
            self.volume_start_times = volume_start_times

    def execute(self):

        iterable = parcels(self.rho, exclude=[np.inf])
        def model_params(i):
            return (np.asarray(self.rho)[i].mean(),)
        m = model_generator(self.formula, self.data,
                            self.volume_start_times,
                            iterable=iterable,
                            model_type=ARModel,
                            model_params=model_params)
        r = results_generator(m)

        def reshape(i, x):
            """
            To write output, arrays have to be reshaped --
            this function does the appropriate reshaping for the two
            passes of fMRIstat.

            These passes are:
              i) 'slices through the z-axis'
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

        o = generate_output(self.outputs, r, reshape=reshape)


def output_T(outbase, contrast, fmri_image, effect=True, sd=True, t=True,
             clobber=False):
    """
    Parameters
    ----------
    outbase : string
        
        Base filename that will be used to construct a set of files
        for the TContrast.  For example, outbase='output.nii' will
        result in the following files (assuming defaults for all other
        params): output_effect.nii, output_sd.nii, output_t.nii

    contrast : a TContrast

    """

    def build_filename(label):
        index = outbase.find('.')
        return ''.join([outbase[:index], '_', label, outbase[index:]])

    if effect:
        effectim = ModelOutputImage(build_filename('effect'),
                                    fmri_image[0].coordmap, 
                                    fmri_image[0].shape, clobber=clobber)
    else:
        effectim = None

    if sd:
        sdim = ModelOutputImage(build_filename('sd'),
                                fmri_image[0].coordmap, fmri_image[0].shape, 
                                clobber=clobber)
    else:
        sdim = None

    if t:
        tim = ModelOutputImage(build_filename('t'),
                               fmri_image[0].coordmap,fmri_image[0].shape, 
                               clobber=clobber)
    else:
        tim = None
    return regression.TOutput(contrast, effect=effectim, sd=sdim, t=tim)


def output_F(outfile, contrast, fmri_image, clobber=False):
    f = ModelOutputImage(outfile, fmri_image[0].coordmap, fmri_image[0].shape, 
                         clobber=clobber)
    return regression.RegressionOutput(f, lambda x: 
                                       regression.output_F(x, contrast))

                             
def output_AR1(outfile, fmri_image, clobber=False):
    """
    Create an output file of the AR1 parameter from the OLS pass of
    fmristat.

    image: FmriImageList 

    """
    outim = ModelOutputImage(outfile, fmri_image[0].coordmap, 
                             fmri_image[0].shape, clobber=clobber)
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
        T[1:,1:] = fmri_image[0].affine
        T[0,0] = (fmri_image.volume_start_times[1:] - 
                  fmri_image.volume_start_times[:-1]).mean()
        # FIXME: NIFTI specific naming here
        innames = ["l"] + list(g.input_coords.coord_names)
        outnames = ["t"] + list(g.output_coords.coord_names)
        cmap = Affine.from_params(innames,
                                  outnames, T)
        shape = (n,) + fmri_image[0].shape
    elif isinstance(fmri_image, Image):
        cmap = fmri_image.coordmap
        shape = fmri_image.shape
    else:
        raise ValueError, "expecting FmriImageList or 4d Image"

    outim = ModelOutputImage(outfile, cmap, shape, clobber=clobber)
    return regression.RegressionOutput(outim, regression.output_resid)


def generate_output(outputs, iterable, reshape=lambda x, y: (x, y)):
    """
    Write out results of a given output.

    In the regression setting, results is generally
    going to be a scipy.stats.models.model.LikelihoodModelResults instance.
    """
    for i, results in iterable:
        for output in outputs:
            if not hasattr(output, "list"): # lame test here
                k, d = reshape(i, output(results))
                output[k] = d
            else:
                r = output(results)
                for j, l in enumerate(output.list):
                    k, d = reshape(i, r[j])
                    l[k] = d

    # flush outputs, if necessary

    for output in outputs:
        if isinstance(output, regression.RegressionOutput):
            if hasattr(output.img, 'save'):
                output.img.save()
        elif isinstance(output, regression.RegressionOutputList):
            for im in output.list:
                if hasattr(im, 'save'):
                    im.save()
