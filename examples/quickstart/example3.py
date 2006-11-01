import numpy as N
import pylab

from neuroimaging.core.image.image import Image
from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils.tests.data import repository

def mask_and_func(subject=0, run=1, offset=5):
    M = Image('FIAC/fiac%d/fonc%d/fsl/mask.img' % (subject,run), repository)
    m = N.zeros(M.grid.shape)
    middle = [slice(i/2-offset, i/2+offset, 1) for i in m.shape]
    m[middle] = 1
    other = [slice(i/2-2*offset, i/2-offset, 1) for i in m.shape]
    m[other] = 2
    mm = Image(m, grid=M.grid)
    f = fMRIImage('FIAC/fiac%d/fonc%d/fsl/filtered_func_data.img' % (subject, run), datasource=repository)
    return mm, f

m, f = mask_and_func()
f.grid.set_iter_param("itertype", 'parcel')
f.grid.parcelmap = m[:]
f.grid.parcelseq = [1, 2]

means = {}
for d in f:
    means[f.label] = N.mean(d, axis=1)

f.grid.parcelseq = [0, 2] # changing the parcelseq to select "regions"
for d in f:
    print d.shape

pylab.plot(means[(1,)], means[(2,)], 'bo')
pylab.show()
