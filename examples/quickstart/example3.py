import neuroimaging as ni
import numpy as N
import pylab

def mask_and_func(subject=0, run=1, offset=5):
    M = ni.image.Image('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/mask.img' % (subject,run))    
    m = N.zeros(M.grid.shape, N.Float)
    middle = [slice(i/2-offset, i/2+offset, 1) for i in m.shape]
    m[middle] = 1
    mm = ni.image.Image(m, grid=M.grid)
    f = ni.fmri.fMRIImage('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/filtered_func_data.img' % (subject, run))
    return mm, f

m, f = mask_and_func()
f.grid.itertype = 'parcel'
f.grid.labels = m
f.grid.labelset = [0.,1.]

print m.image.data.sum()
for d in f:
    print d.shape
