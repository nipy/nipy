# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module defines the two default GLM passes of fmristat

The results of both passes of the GLM get pushed around by generators, which
know how to get out the (probably 3D) data for each slice, or parcel (for the
AR) case, estimate in 2D, then store the data back again in its original shape.

The containers here, in the execute methods, know how to reshape the data on the
way into the estimation (to 2D), then back again, to 3D, or 4D.

It's relatively easy to do this when just iterating over simple slices, but it
gets a bit more complicated when taking arbitrary shaped samples from the image,
as we do for estimating the AR coefficients, where we take all the voxels with
similar AR coefficients at once.
"""
from __future__ import absolute_import

import copy

import os.path as path

import numpy as np
import numpy.linalg as npl

from nipy.algorithms.statistics.models.regression import (
    OLSModel, ARModel, ar_bias_corrector, ar_bias_correct)
from nipy.algorithms.statistics.formula import make_recarray

# nipy core imports
from nipy.core.api import Image, parcels, matrix_generator, AffineTransform

# nipy IO imports
from nipy.io.api import save_image

# fmri imports
from ..api import FmriImageList, axis0_generator

from . import outputters


class ModelOutputImage(object):
    """
    These images have their values filled in as the model is fit, and
    are saved to disk after being completely filled in.

    They are saved to disk by calling the 'save' method.

    The __getitem__ and __setitem__ calls are delegated to a private
    Image.  An exception is raised if trying to get/set data after the
    data has been saved to disk.
    """

    def __init__(self, filename, coordmap, shape, clobber=False):
        self.filename = filename
        self._im_data = np.zeros(shape)
        self._im = Image(self._im_data, coordmap)
        # Using a dangerous undocumented API here
        self.clobber = clobber
        self._flushed = False

    def save(self):
        """
        Save current Image data to disk
        """
        if not self.clobber and path.exists(self.filename):
            raise ValueError('trying to clobber existing file')
        save_image(self._im, self.filename)
        self._flushed = True
        del(self._im)

    def __getitem__(self, item):
        if self._flushed:
            raise ValueError('trying to read value from a '
                             'saved ModelOutputImage')
        return self._im_data[item]

    def __setitem__(self, item, value):
        if self._flushed:
            raise ValueError('trying to set value on saved'
                             'ModelOutputImage')
        self._im_data[item] = value


def model_generator(formula, data, volume_start_times, iterable=None,
                    slicetimes=None, model_type=OLSModel,
                    model_params = lambda x: ()):
    """
    Generator for the models for a pass of fmristat analysis.
    """
    volume_start_times = make_recarray(volume_start_times.astype(float), 't')
    # Generator for slices of the data with time as first axis
    axis0_gen = axis0_generator(data, slicers=iterable)
    # Iterate over 2D slices of the data
    for indexer, indexed_data in matrix_generator(axis0_gen):
        model_args = model_params(indexer) # model may depend on i
        # Get the design for these volume start times
        design = formula.design(volume_start_times, return_float=True)
        # Make the model from the design
        rmodel = model_type(design, *model_args)
        yield indexer, indexed_data, rmodel


def results_generator(model_iterable):
    """
    Generator for results from an iterator that returns
    (index, data, model) tuples.

    See model_generator.
    """
    for i, d, m in model_iterable:
        yield i, m.fit(d)


class OLS(object):
    """
    First pass through fmri_image.

    Parameters
    ----------
    fmri_image : `FmriImageList` or 4D image
       object returning 4D data from np.asarray, with first
       (``object[0]``) axis being the independent variable of the model;
       object[0] returns an object with attribute ``shape``.
    formula :  :class:`nipy.algorithms.statistics.formula.Formula`
    outputs :
    volume_start_times :
    """

    def __init__(self, fmri_image, formula, outputs=[],
                 volume_start_times=None):
        self.fmri_image = fmri_image
        try:
            self.data = fmri_image.get_data()
        except AttributeError:
            self.data = fmri_image.get_list_data(axis=0)
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

    Parameters
    ----------
    resid:  array-like
        residuals from model
    model:  an OLS model used to estimate residuals

    Returns
    -------
    output : array
        shape (order, resid
    """
    invM = ar_bias_corrector(design, npl.pinv(design), order)
    return ar_bias_correct(resid, order, invM)


class AR1(object):
    """
    Second pass through fmri_image.

    Parameters
    ----------
    fmri_image : `FmriImageList`
       object returning 4D array from ``np.asarray``, having attribute
       ``volume_start_times`` (if `volume_start_times` is None), and
       such that ``object[0]`` returns something with attributes ``shape``
    formula :  :class:`nipy.algorithms.statistics.formula.Formula`
    rho : ``Image``
       image of AR(1) coefficients.  Returning data from
       ``rho.get_data()``, and having attribute ``coordmap``
    outputs :
    volume_start_times : 
    """

    def __init__(self, fmri_image, formula, rho, outputs=[],
                 volume_start_times=None):
        self.fmri_image = fmri_image
        try:
            self.data = fmri_image.get_data()
        except AttributeError:
            self.data = fmri_image.get_list_data(axis=0)
        self.formula = formula
        self.outputs = outputs
        # Cleanup rho values, truncate them to a scale of 0.01
        g = copy.copy(rho.coordmap)
        rho = rho.get_data()
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
            return (self.rho.get_data()[i].mean(),)
        # Generates indexer, data, model
        m = model_generator(self.formula, self.data,
                            self.volume_start_times,
                            iterable=iterable,
                            model_type=ARModel,
                            model_params=model_params)
        # Generates indexer, data, 2D results
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
            if len(x.shape) == 2: # 2D imput matrix
                if type(i) is type(1): # integer indexing
                    # reshape to ND (where N is probably 4)
                    x.shape = (x.shape[0],) + self.fmri_image[0].shape[1:]
                # Convert lists to tuples, put anything else into a tuple
                if type(i) not in [type([]), type(())]:
                    i = (i,)
                else:
                    i = tuple(i)
                # Add : to indexing
                i = (slice(None,None,None),) + tuple(i)
            else: # not 2D
                if type(i) is type(1): # integer indexing
                    x.shape = self.fmri_image[0].shape[1:]
            return i, x

        # Put results pulled from results generator r, into outputs
        o = generate_output(self.outputs, r, reshape=reshape)


def output_T(outbase, contrast, fmri_image, effect=True, sd=True, t=True,
             clobber=False):
    """ Return t contrast regression outputs list for `contrast`

    Parameters
    ----------
    outbase : string
        Base filename that will be used to construct a set of files
        for the TContrast.  For example, outbase='output.nii' will
        result in the following files (assuming defaults for all other
        params): output_effect.nii, output_sd.nii, output_t.nii
    contrast : array
        F contrast matrix
    fmri_image : ``FmriImageList`` or ``Image``
        object such that ``object[0]`` has attributes ``shape`` and
        ``coordmap``
    effect : {True, False}, optional
        whether to write an effect image
    sd : {True, False}, optional
        whether to write a standard deviation image
    t : {True, False}, optional
        whether to write a t image
    clobber : {False, True}, optional
        whether to overwrite images that exist.

    Returns
    -------
    reglist : ``RegressionOutputList`` instance
        Regression output list with selected outputs, where selection is by
        inputs `effect`, `sd` and `t`

    Notes
    -----
    Note that this routine uses the corresponding ``output_T`` routine in
    :mod:`outputters`, but indirectly via the TOutput object.
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
    return outputters.TOutput(contrast, effect=effectim, sd=sdim, t=tim)


def output_F(outfile, contrast, fmri_image, clobber=False):
    ''' output F statistic images

    Parameters
    ----------
    outfile : str
        filename for F contrast image
    contrast : array
        F contrast matrix
    fmri_image : ``FmriImageList`` or ``Image``
        object such that ``object[0]`` has attributes ``shape`` and
        ``coordmap``
    clobber : bool
        if True, overwrites previous output; if False, raises error

    Returns
    -------
    f_reg_out : ``RegressionOutput`` instance
        Object that can a) be called with a results instance as argument,
        returning an array, and b) accept the output array for storing, via
        ``obj[slice_spec] = arr`` type slicing.
    '''
    f = ModelOutputImage(outfile, fmri_image[0].coordmap, fmri_image[0].shape,
                         clobber=clobber)
    return outputters.RegressionOutput(f, lambda x:
                                       outputters.output_F(x, contrast))


def output_AR1(outfile, fmri_image, clobber=False):
    """
    Create an output file of the AR1 parameter from the OLS pass of
    fmristat.

    Parameters
    ----------
    outfile :
    fmri_image : ``FmriImageList`` or 4D image
       object such that ``object[0]`` has attributes ``coordmap`` and ``shape``
    clobber : bool
       if True, overwrite previous output

    Returns
    -------
    regression_output : ``RegressionOutput`` instance
    """
    outim = ModelOutputImage(outfile, fmri_image[0].coordmap,
                             fmri_image[0].shape, clobber=clobber)
    return outputters.RegressionOutput(outim, outputters.output_AR1)


def output_resid(outfile, fmri_image, clobber=False):
    """
    Create an output file of the residuals parameter from the OLS pass of
    fmristat.

    Uses affine part of the first image to output resids unless
    fmri_image is an Image.

    Parameters
    ----------
    outfile :
    fmri_image : ``FmriImageList`` or 4D image
       If ``FmriImageList``, needs attributes ``volume_start_times``,
       supports len(), and object[0] has attributes ``affine``,
       ``coordmap`` and ``shape``, from which we create a new 4D
       coordmap and shape
       If 4D image, use the images coordmap and shape
    clobber : bool
       if True, overwrite previous output

    Returns
    -------
    regression_output :
    """
    if isinstance(fmri_image, FmriImageList):
        n = len(fmri_image.list)
        T = np.zeros((5,5))
        g = fmri_image[0].coordmap
        T[1:,1:] = fmri_image[0].affine
        T[0,0] = (fmri_image.volume_start_times[1:] -
                  fmri_image.volume_start_times[:-1]).mean()
        # FIXME: NIFTI specific naming here
        innames = ["t"] + list(g.function_domain.coord_names)
        outnames = ["t"] + list(g.function_range.coord_names)
        cmap = AffineTransform.from_params(innames, outnames, T)
        shape = (n,) + fmri_image[0].shape
    elif isinstance(fmri_image, Image):
        cmap = fmri_image.coordmap
        shape = fmri_image.shape
    else:
        raise ValueError("expecting FmriImageList or 4d Image")

    outim = ModelOutputImage(outfile, cmap, shape, clobber=clobber)
    return outputters.RegressionOutput(outim, outputters.output_resid)


def generate_output(outputs, iterable, reshape=lambda x, y: (x, y)):
    """
    Write out results of a given output.

    In the regression setting, results is generally going to be a
    scipy.stats.models.model.LikelihoodModelResults instance.

    Parameters
    ----------
    outputs : sequence
        sequence of output objects
    iterable : object
        Object which iterates, returning tuples of (indexer, results), where
        ``indexer`` can be used to index into the `outputs`
    reshape : callable
        accepts two arguments, first is the indexer, and the second is the array
        which will be indexed; returns modified indexer and array ready for
        slicing with modified indexer.
    """
    for indexer, results in iterable:
        for output in outputs:
            # Might be regression output object
            if not hasattr(output, "list"): # lame test here
                k, d = reshape(indexer, output(results))
                output[k] = d
            else:
                # or a regression output list (like a TOutput, with several
                # images to output to)
                r = output(results)
                for j, l in enumerate(output.list):
                    k, d = reshape(indexer, r[j])
                    l[k] = d
    # flush outputs, if necessary
    for output in outputs:
        if isinstance(output, outputters.RegressionOutput):
            if hasattr(output.img, 'save'):
                output.img.save()
        elif isinstance(output, outputters.RegressionOutputList):
            for im in output.list:
                if hasattr(im, 'save'):
                    im.save()
