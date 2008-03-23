#!/bin/env python
'''
Show PCA series from example functional image

Usage:
pca_example.py
'''

import numpy as np
import pylab

from neuroimaging.core.api import Image, load_image
from neuroimaging.modalities.fmri.api import FmriImage 
from neuroimaging.modalities.fmri.pca import PCAmontage
from neuroimaging.utils.tests.data import repository

# Load an fMRI image
fmridata = load_image("test_fmri.hdr", datasource=repository)

# Create a mask
frame = fmridata.frame(0)
mask = Image(np.greater(frame[:], 500).astype(np.float64), frame.grid)

# Fit PCAmontage which allows you to visualize the results
pca_montage = PCAmontage(fmridata, mask=mask)
pca_montage.fit()

# Return calculated output PCA images into image object list
# Write images to disk
output = pca_montage.images(which=range(4))
for i, img in enumerate(output):
    fname = 'pca_component_%04d.nii' % i
    img.tofile(fname, clobber=True)
    
# View the results
# compare with "http://www.math.mcgill.ca/keith/fmristat/figs/figpca1.jpg"
pca_montage.time_series()
pca_montage.montage()
pylab.show()
