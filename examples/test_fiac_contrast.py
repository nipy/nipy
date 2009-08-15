#!/usr/bin/env python 
"""
This script requires the nipy-data package to run. It is an example of
using a general linear model in single-subject fMRI data analysis
context. 

Usage: 
  python test_fiac_contrast [contrast_vector]

  contrast_vector is input as a list, for instance: [1,-1,1,-1]

Author: Alexis Roche, 2009. 
"""

from nipy.neurospin.statistical_mapping import LinearModel
from nipy.io.imageformats import load as load_image, save as save_image
from nipy.utils import example_data

import numpy as np
import sys

# Optinal argument
cvect = [1,0,0,0]
if len(sys.argv)>1: 
    tmp = list(sys.argv[1])
    tmp.remove('[')
    tmp.remove(']')
    for i in range(tmp.count(',')):
        tmp.remove(',')
    cvect = map(float, tmp)

# Fetch data
fmri_dataset_path = example_data.get_filename('fiac','fiac0','rarun1.nii.gz')
design_matrix_path = example_data.get_filename('fiac','fiac0','run1_mat.npz')
 
# Get design matrix as numpy array
print('Loading design matrix...')
X = np.load(design_matrix_path)['X']

# Get fMRI data as numpy array
print('Loading fmri data...')
Y = load_image(fmri_dataset_path)

# GLM options
model = 'ar1'
##model = 'spherical'
    
# GLM fitting 
# Note that it is possible to pass a Mask argument 
print('starting fit...')
glm = LinearModel(Y, X, mask=None, model=model)

# Compute the required contrast
print('Computing test contrast image...')
c = np.zeros(X.shape[1])
c[0:4] = cvect
con, vcon, zmap, dof = glm.contrast(c)

##save_image(con, whatever_filename_you_like)
