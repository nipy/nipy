#!/usr/bin/env python 
"""
This script requires the nipy-data package to run. It is an example of
simultaneous motion correction and slice timing correction in
multi-session fMRI data from the FIAC 2005 dataset. Specifically, it
uses the first two sessions of subject 'fiac0'. 

Usage: 
  python space_time_realign [iterations]

  where iterations is a positive integer (set to 1 by default). If
  zero, no motion correction is performed and the data is only
  corrected for slice timing. The larger iterations, the more accurate
  motion correction.

Two images will be created in the working directory for the realigned
series:

rarun1.nii
rarun2.nii

Author: Alexis Roche, 2009. 
"""

from nipy.io.imageformats import load as load_image, save as save_image
from nipy.neurospin.image_registration import image4d, realign4d, resample4d
from nipy.utils import example_data

from os.path import join, split
import sys


# Optinal argument
iterations = 1
if len(sys.argv)>1: 
    iterations = int(sys.argv[1])

# Input images are provided with the nipy-data package
runs = ['run1', 'run2']
runnames = [example_data.get_filename('fiac','fiac0',run+'.nii.gz') \
                for run in runs]
images = [load_image(runname) for runname in runnames]

# Create Image4d instances -- this is a local class representing a
# series of 3d images
runs = [image4d(im, tr=2.5, slice_order='ascending', interleaved=True) \
            for im in images]

# Correct motion within- and between-sessions
# By default 2 within-loops and 5 between-loops will be performed
transforms = realign4d(runs, within_loops=iterations, between_loops=2*iterations)

# Resample data on a regular space+time lattice using 4d interpolation
corr_runs = [resample4d(runs[i], transforms=transforms[i]) for i in range(len(runs))]

# Save images 
for i in range(len(runs)):
    aux = split(runnames[i])
    save_image(corr_runs[i], join(aux[0], 'ra'+aux[1]))


