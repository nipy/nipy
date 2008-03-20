import numpy as np
from numpy.random import standard_normal as noise

from neuroimaging.testing import funcfile, anatfile
from neuroimaging.core.api import load_image
from neuroimaging.modalities.fmri.api import fromimage

from neuroimaging.fixes.scipy.stats_models.regression import OLSModel as ols_model

fi = fromimage(load_image(funcfile))

design = noise((len(fi.list),3))
model = ols_model(design)

print fi[0].grid.ndim, 'what'
print fi[0].affine.shape
print fi[0][1]

def fit(input):
    return model.fit(input).resid

def g(iterable):
    l = len(list(fi.list)) # this may require generating some large list --
                           # do we assume fi.list has an implicit len?
                           # in this case, we could write len(fi.list) above
    for i in iterable:

        # for a regular 4d image this call above does not have much overhead
        # but it is more than fi[:,i] when you know that fi has an array
        # associated with it

        indata = np.asarray([np.asarray(fi[j])[i] for j in range(l)])
        indata.shape = (indata.shape[0], np.product(indata.shape[1:]))

        # this probably makes a copy, in general
        #
        # unless we know that FmriImage came from an underlying 4d array,
        # I think the overhead above is unavoidable, 
        # even if we hide this line in a method of FmriImage
        #

        resid = fit(indata) 
        yield resid.shape

for f in g(range(fi[0].shape[0])):
    print f

a = np.asarray(fi[0])
b = [np.greater(a, a.mean()), np.less_equal(a, a.mean())]

for f in g(b):
    print f


    
