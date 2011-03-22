#!/usr/bin/env python 
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This script requires the nipy-data package to run. It is an example of
using a general linear model in single-subject fMRI data analysis
context. Two sessions of the same subject are taken from the FIAC'05
dataset.

Usage: 
  python compute_fmri_contrast [contrast_vector]

Example: 
  python compute_fmri_contrast [1,-1,1,-1]

  An image file called zmap.nii.gz will be created in the working
  directory.

Author: Alexis Roche, 2009. 
"""
import numpy as np
import sys
from nibabel import load as load_image, save as save_image

from nipy.labs.statistical_mapping import LinearModel
from nipy.utils import example_data

# Optional argument
cvect = [1,0,0,0]
if len(sys.argv)>1: 
    tmp = list(sys.argv[1])
    tmp.remove('[')
    tmp.remove(']')
    for i in range(tmp.count(',')):
        tmp.remove(',')
    cvect = map(float, tmp)


# Input files
fmri_files = [example_data.get_filename('fiac','fiac0',run) for run in ['run1.nii.gz','run2.nii.gz']]
design_files = [example_data.get_filename('fiac','fiac0',run) for run in ['run1_design.npz','run2_design.npz']]
mask_file = example_data.get_filename('fiac','fiac0','mask.nii.gz') 

# Get design matrix as numpy array
print('Loading design matrices...')
X = [np.load(f)['X'] for f in design_files]

# Get multi-session fMRI data 
print('Loading fmri data...')
Y = [load_image(f) for f in fmri_files]

# Get mask image
print('Loading mask...')
mask = load_image(mask_file)

# GLM fitting 
print('Starting fit...')
##glm = LinearModel(Y, X, mask=mask, model='ar1')
glm = LinearModel(Y, X, mask=mask)

# Compute the required contrast
print('Computing test contrast image...')
nregressors = X[0].shape[1] ## should check that all design matrices have the same 
c = np.zeros(nregressors)
c[0:4] = cvect
con, vcon, zmap, dof = glm.contrast(c)

# Save Zmap image 
save_image(zmap, 'zmap.nii.gz')
