import neuroimaging as ni
import numpy as N
import pylab

def mask_and_func(subject=0, run=1, offset=5):
    M = ni.image.Image('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/mask.img' % (subject,run))    
    m = N.zeros(M.grid.shape)
    middle = [slice(i/2-offset, i/2+offset, 1) for i in m.shape]
    m[middle] = 1
    f = ni.fmri.fMRIImage('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/filtered_func_data.img' % (subject, run))
    return m, f

m, f = mask_and_func()
a = N.zeros(f.grid.shape[0])
nvox = m.sum()

for i in range(a.shape[0]):
    a[i] = (m * f.getslice(slice(i,i+1))).sum() / nvox

pylab.plot(a)
pylab.show()
