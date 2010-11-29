#!/usr/bin/env python 
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

from nipy.algorithms.registration import FmriRealign4d

from nipy import load_image, save_image
from nipy.utils import example_data

from os.path import join, split
import sys


# Optional argument
loops = 1 
if len(sys.argv)>1: 
    loops = int(sys.argv[1])

# Input images are provided with the nipy-data package
runnames = [example_data.get_filename('fiac','fiac0',run+'.nii.gz') \
                for run in ('run1','run2')]
runs = [load_image(run) for run in runnames]

# Spatio-temporal realigner
R = FmriRealign4d(runs, tr=2.5, slice_order='ascending', interleaved=True)

# Estimate motion within- and between-sessions
R.estimate(loops=loops)

# Resample data on a regular space+time lattice using 4d interpolation
corr_runs = R.resample()

# Save images 
for i in range(len(runs)):
    aux = split(runnames[i])
    save_image(corr_runs[i], join('ra'+aux[1]))


