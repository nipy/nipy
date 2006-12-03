#!/bin/env python
'''
Show PCA series from example functional image

Usage:
pca_example.py
'''

import numpy as N
import pylab

from neuroimaging.core.image.image import Image
from neuroimaging.modalities.fmri import fMRIImage 
from neuroimaging.modalities.fmri.pca import PCAmontage
from neuroimaging.utils.tests.data import repository

# Load an fMRI image
fmridata = fMRIImage("test_fmri.hdr", datasource=repository)

# Create a mask
frame = fmridata.frame(0)
mask = Image(N.greater(frame[:], 500).astype(N.float64), grid=frame.grid)

# Fit PCAmontage which allows you to visualize the results
p = PCAmontage(fmridata, mask=mask)
p.fit()

# Return calculated output PCA images into image object list
# Write images to disk
output = p.images(which=range(4))
for i, img in enumerate(output):
    fname = 'pca_component_%04d.nii' % i
    img.tofile(fname, clobber=True)
    
# View the results
# compare with "http://www.math.mcgill.ca/keith/fmristat/figs/figpca1.jpg"
p.time_series()
p.montage()
pylab.show()
