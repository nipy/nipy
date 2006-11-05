#!/bin/env python
'''
Plot the mean intensity of the middle slice within the brain (as
defined by the mask) across time for the first subject and first run
in the example dataset

Usage
middle_time_series.py
'''

import numpy as N
import pylab

from neuroimaging.core.image.image import Image
from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils.tests.data import repository

def mask_and_func(subject=0, run=1, offset=5):
    ''' Return a slice definition for the masked middle slice of the functional run
    '''
    M = Image('FIAC/fiac%d/fonc%d/fsl/mask.img' % (subject,run), repository)    
    m = N.zeros(M.grid.shape)
    middle = [slice(i/2-offset, i/2+offset, 1) for i in m.shape]
    m[middle] = 1
    f = fMRIImage('FIAC/fiac%d/fonc%d/fsl/filtered_func_data.img' % (subject, run), datasource=repository)
    return m, f

m, f = mask_and_func()
a = N.zeros(f.grid.shape[0])
nvox = m.sum()

for i in range(a.shape[0]):
    a[i] = (m * f[slice(i,i+1)]).sum() / nvox

pylab.plot(a)
pylab.show()
