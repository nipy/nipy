import numpy as np
from numpy.random import standard_normal as noise

from neuroimaging.testing import funcfile, anatfile
from neuroimaging.core.api import load_image
from neuroimaging.modalities.fmri.api import fromimage, fmri_generator
from neuroimaging.core.image.generators import *
from neuroimaging.fixes.scipy.stats.models.regression import OLSModel as ols_model

fd = np.asarray(load_image(funcfile))
fi = fromimage(load_image(funcfile)) # I think it makes more
                                     # sense to use fd instead of fi
                                     # for GLM purposes -- reduces some
                                     # noticeable overhead in creating
                                     # the array from FmriImage.list

# create a design matrix, model and contrast matrix

design = noise((fd.shape[0],3))
model = ols_model(design)
cmatrix = np.array([[1,0,0],[0,1,0]])

# two prototypical functions in a GLM analysis

def fit(input):
    return model.fit(input).resid

def contrast(results):
    return results.Fcontrast(cmatrix)

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
        
"""
Fit a model, iterating over the slices of an array
associated to an FmriImage.
"""

c = np.zeros(fd.shape[1:]) + 0.5
write_data(c, unflatten_generator(contrast_generator(result_generator(flatten_generator(fmri_generator(fd))))))

"""
Fit a model, iterating over the array associated to an FmriImage,
iterating over a list of ROIs defined by binary regions
of the same shape as a frame of FmriImage
"""

a = np.asarray(fd[0]) # this might really be an anatomical image or
                      # AR(1) coefficients 

p = np.greater(a, a.mean())

d = np.ones(fd.shape[1:]) * 2.
write_data(d, contrast_generator(result_generator(flatten_generator(fmri_generator(fd, parcels(p))))))

assert np.allclose(d, c)

e = np.zeros(fd.shape[1:]) + 3.
write_data(e, f_generator(contrast, result_generator(flatten_generator(fmri_generator(fd, parcels(p))))))

assert np.allclose(d, e)

    
