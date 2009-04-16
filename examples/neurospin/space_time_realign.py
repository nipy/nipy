from nipy.neurospin import register 

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

# Create TimeSeries instances -- this is a local class representing a
# series of 3d images
run1 = register.TimeSeries(im1.get_data(), toworld=im1.get_affine(), tr=2.5, 
                           slice_order='ascending', interleaved=True)
run2 = register.TimeSeries(im2.get_data(), toworld=im2.get_affine(), tr=2.5, 
                           slice_order='ascending', interleaved=True)

# Correct motion within- and between-sessions
transforms = register.realign4d([run1, run2]) 

# Resample data on a regular space+time lattice using 4d interpolation
corr_im1 = v.nifti1.Nifti1Image(affine=im1.get_affine(), 
                                data=register.resample4d(run1, transforms=transforms[0]))
corr_im2 = v.nifti1.Nifti1Image(affine=im2.get_affine(), 
                                data=register.resample4d(run2, transforms=transforms[1]))

# Save images 
corr_im1.to_files('corr_run1.nii')
corr_im2.to_files('corr_run2.nii')
