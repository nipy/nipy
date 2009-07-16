import numpy as np
from os.path import join, split
import sys
import time
from glob import glob 

from nipy.io.imageformats import load as load_image, save as save_image
from nipy.neurospin.image_registration import image4d, realign4d, resample4d

# Create Nifti1Image instances from both input files
rootpath = 'D:\\home\\AR203069\\data\\karla'
runnames = glob(join(rootpath, 'fms*.nii'))
print runnames
images = [load_image(rname) for rname in runnames]

"""
#Single-session case
run = image4d(images[0], tr=2.4, slice_order='ascending', interleaved=False)
transforms = realign4d(run)
corr_run = resample4d(run, transforms=transforms)
save_image(corr_run, 'corr_run1.nii')
"""

# Multi-session case
# Create Image4d instances -- this is a local class representing a
# series of 3d images
runs = [image4d(im, tr=2.4, slice_order='ascending', interleaved=False) 
            for im in images]

# To reduce the sequence to two runs, just do: 
##runs = runs[0:2]

# Correct motion within- and between-sessions
transforms = realign4d(runs)

#By default 2 within-loops and 5 between-loops will be performed
#To reduce that, input: 
##transforms = realign4d(runs, within_loops=1, between_loops=4)

# Resample data on a regular space+time lattice using 4d interpolation
corr_runs = [resample4d(runs[i], transforms=transforms[i]) for i in range(len(runs))]

# Save images 
for i in range(len(runs)):
    aux = split(runnames[i])
    save_image(corr_runs[i], join(aux[0], 'ra'+aux[1]))


