import numpy as np
from os.path import join, split
import sys
import time
from glob import glob 

from nipy.neurospin.image_registration import image4d, realign4d, resample4d
import nifti as nf

# Create Nifti1Image instances from both input files
rootpath = 'D:\\home\\AR203069\\data\\karla'
runnames = glob(join(rootpath, '*.nii'))
print runnames
images = [nf.load(rname) for rname in runnames]

"""
#Single-session case
run = image4d(images[0], tr=2.4, slice_order='ascending', interleaved=False)
transforms = realign4d(run)
corr_run = resample4d(run, transforms=transforms)
corr_img = nf.nifti1.Nifti1Image(corr_run.get_data(), corr_run.get_affine())
nf.save(corr_img, 'corr_run1.nii')
"""

# Multi-session case
# Create Image4d instances -- this is a local class representing a
# series of 3d images
runs = [image4d(im, tr=2.4, slice_order='ascending', interleaved=False) for im in images]
##runs = runs[0:2]

# Correct motion within- and between-sessions
transforms = realign4d(runs)

# Resample data on a regular space+time lattice using 4d interpolation
corr_runs = [resample4d(runs[i], transforms=transforms[i]) for i in range(len(runs))]
corr_images = [nf.nifti1.Nifti1Image(run.get_data(), run.get_affine()) for run in corr_runs]

# Save images 
for i in range(len(runs)):
    aux = split(runnames[i])
    nf.save(corr_images[i], join(aux[0], 'ra'+aux[1]))


