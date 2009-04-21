import numpy as np
from os.path import join
import sys
import time
from glob import glob 

from nipy.neurospin.image_registration import image4d, realign4d, resample4d

# Use Matthew's volumeimages for I/O. 
import volumeimages as v

# Create Nifti1Image instances from both input files
rootpath = 'D:\\data\\karla'
runnames = glob(join(rootpath, '*.nii'))
### HACK 
runnames = runnames[0:2]
print runnames
images = [v.load(rname) for rname in runnames]

# Create Image4d instances -- this is a local class representing a
# series of 3d images
"""
Pour l'ordre d'acquisition, il s'agit de sequential - ascending et
pour l'enfant qui a le plus bougé, il s'agit de ms070149.
Le TR est de 2,4. 
"""
runs = [image4d(im, tr=2.4, slice_order='ascending', interleaved=False) for im in images]

# Correct motion within- and between-sessions
transforms = realign4d(runs)

# Resample data on a regular space+time lattice using 4d interpolation
corr_runs = [resample4d(runs[i], transforms=transforms[i]) for i in range(len(runs))]
corr_images = [v.nifti1.Nifti1Image(run.get_data(), run.get_affine()) for run in corr_runs]

# Save images 
for i in range(len(runs)):
    v.save(corr_images[i], 'corr_run'+str(i)+'.nii')
