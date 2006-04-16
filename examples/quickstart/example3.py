import neuroimaging as ni
import numpy as N
import pylab

def mask_and_func(subject=0, run=1, offset=5):
    M = ni.image.Image('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/mask.img' % (subject,run))    
    m = N.zeros(M.grid.shape)
    middle = [slice(i/2-offset, i/2+offset, 1) for i in m.shape]
    m[middle] = 1
    other = [slice(i/2-offset+1, i/2+offset+1, 1) for i in m.shape]
    m[other] = 2
    mm = ni.image.Image(m, grid=M.grid)
    f = ni.fmri.fMRIImage('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/filtered_func_data.img' % (subject, run))
    return mm, f

m, f = mask_and_func()
f.grid.itertype = 'parcel'
f.grid.labels = m.readall()
f.grid.labelset = [1, 2]

means = {}
for d in f:
    means[f.label] = N.mean(d, axis=1)

pylab.plot(means[1], means[2], 'bo')
pylab.show()
