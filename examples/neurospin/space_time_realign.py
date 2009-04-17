from nipy.neurospin.image_registration import image4d, realign4d, resample4d

# Use Matthew's volumeimages for I/O. 
import volumeimages as v

import numpy as np
from os.path import join
import sys
import time

rootpath = 'D:\\data\\fiac'
runnames = ['run1', 'run1']

# Create Nifti1Image instances from both input files
im1 = v.load(join(rootpath, runnames[0]+'.nii'))
im2 = v.load(join(rootpath, runnames[1]+'.nii'))

# Create Image4d instances -- this is a local class representing a
# series of 3d images
run1 = image4d(im1, tr=2.5, slice_order='ascending', interleaved=True)
run2 = image4d(im2, tr=2.5, slice_order='ascending', interleaved=True)

# Correct motion within- and between-sessions
transforms = realign4d([run1, run2]) 

# Resample data on a regular space+time lattice using 4d interpolation
corr_run1 = resample4d(run1, transforms=transforms[0])
corr_run2 = resample4d(run2, transforms=transforms[0])
corr_im1 = v.nifti1.Nifti1Image(corr_run1.get_data(), corr_run1.get_affine())
corr_im2 = v.nifti1.Nifti1Image(corr_run2.get_data(), corr_run2.get_affine())

# Save images 
v.save(corr_im1, 'corr_run1.nii')
v.save(corr_im2, 'corr_run2.nii')
