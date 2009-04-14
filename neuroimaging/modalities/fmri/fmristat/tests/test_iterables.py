import warnings
import numpy as np
from numpy.random import standard_normal as noise

from nipy.testing import *
from nipy.io.api import load_image
from nipy.modalities.fmri.api import fromimage, fmri_generator
from nipy.core.image.generators import *
from nipy.fixes.scipy.stats.models.regression import OLSModel as ols_model

def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")

def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


# two prototypical functions in a GLM analysis
def fit(input):
    return model.fit(input).resid

def contrast(results):
    return results.Fcontrast(cmatrix)


# generators
def result_generator(datag):
    for i, fdata in datag:
        yield i, model.fit(fdata)

def flatten_generator(ing):
    for i, r in ing:
        r.shape = (r.shape[0], np.product(r.shape[1:]))
        yield i, r

def unflatten_generator(ing):
    for i, r in ing:
        r.shape = (2,20)
        yield i, r

def contrast_generator(resultg):
    for i, r in resultg:
        yield i, np.asarray(contrast(r))


class TestIters(TestCase):
    def setUp(self):
        self.fd = np.asarray(load_image(funcfile))
        self.fi = fromimage(load_image(funcfile))
        # I think it makes more sense to use fd instead of fi for GLM
        # purposes -- reduces some noticeable overhead in creating the
        # array from FmriImage.list

        # create a design matrix, model and contrast matrix

        self.design = noise((self.fd.shape[0],3))
        self.model = ols_model(self.design)
        self.cmatrix = np.array([[1,0,0],[0,1,0]])

    def test_iterate_over_image(self):
        # Fit a model, iterating over the slices of an array
        # associated to an FmriImage.
        c = np.zeros(self.fd.shape[1:]) + 0.5
        res_gen = result_generator(flatten_generator(fmri_generator(fd)))
        write_data(c, unflatten_generator(contrast_generator(res_gen)))

        # Fit a model, iterating over the array associated to an
        # FmriImage, iterating over a list of ROIs defined by binary
        # regions of the same shape as a frame of FmriImage
        
        # this might really be an anatomical image or AR(1) coefficients 
        a = np.asarray(fd[0]) 
        p = np.greater(a, a.mean())
        d = np.ones(fd.shape[1:]) * 2.0
        flat_gen = flatten_generator(fmri_generator(fd, parcels(p)))
        write_data(d, contrast_generator(result_generator(flat_gen)))

        yield assert_array_almost_equal, d, c

        e = np.zeros(fd.shape[1:]) + 3.0
        flat_gen2 = flatten_generator(fmri_generator(fd, parcels(p)))
        write_data(e, f_generator(contrast, result_generator(flat_gen2)))

        yield assert_array_almost_equal, d, e
