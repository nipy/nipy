# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import warnings
import numpy as np
from numpy.random import standard_normal as noise

from nipy.io.api import load_image
from nipy.core.image.image import rollimg
from nipy.modalities.fmri.api import FmriImageList, axis0_generator
from nipy.core.utils.generators import (write_data, parcels,
                                        f_generator)

from nipy.algorithms.statistics.models.regression import OLSModel

from numpy.testing import assert_array_almost_equal
from nipy.testing import funcfile


def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")


def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


# Module globals
FIMG = load_image(funcfile)
# Put time on first axis
FIMG = rollimg(FIMG, 't')
FDATA = FIMG.get_data()
FIL = FmriImageList.from_image(FIMG)

# I think it makes more sense to use FDATA instead of FIL for GLM
# purposes -- reduces some noticeable overhead in creating the
# array from FmriImageList

# create a design matrix, model and contrast matrix
DESIGN = noise((FDATA.shape[0],3))
MODEL = OLSModel(DESIGN)
CMATRIX = np.array([[1,0,0],[0,1,0]])

# two prototypical functions in a GLM analysis
def fit(input):
    return MODEL.fit(input).resid


def contrast(results):
    return results.Fcontrast(CMATRIX)


# generators
def result_generator(datag):
    for i, fdata in datag:
        yield i, MODEL.fit(fdata)


def flatten_generator(ing):
    for i, r in ing:
        r = r.reshape((r.shape[0], -1))
        yield i, r


def unflatten_generator(ing):
    for i, r in ing:
        r = r.reshape(FIMG.shape[2:])
        yield i, r


def contrast_generator(resultg):
    for i, r in resultg:
        yield i, np.asarray(contrast(r))


def test_iterate_over_image():
    # Fit a model, iterating over the slices of an array
    # associated to an FmriImage.
    c = np.zeros(FDATA.shape[1:]) + 0.5
    res_gen = result_generator(flatten_generator(axis0_generator(FDATA)))
    write_data(c, unflatten_generator(contrast_generator(res_gen)))
    # Fit a model, iterating over the array associated to an
    # FmriImage, iterating over a list of ROIs defined by binary
    # regions of the same shape as a frame of FmriImage

    # this might really be an anatomical image or AR(1) coefficients
    a = np.asarray(FDATA[0])
    p = np.greater(a, a.mean())
    d = np.ones(FDATA.shape[1:]) * 2.0
    flat_gen = flatten_generator(axis0_generator(FDATA, parcels(p)))
    write_data(d, contrast_generator(result_generator(flat_gen)))
    assert_array_almost_equal(d, c)

    e = np.zeros(FDATA.shape[1:]) + 3.0
    flat_gen2 = flatten_generator(axis0_generator(FDATA, parcels(p)))
    write_data(e, f_generator(contrast, result_generator(flat_gen2)))
    assert_array_almost_equal(d, e)
