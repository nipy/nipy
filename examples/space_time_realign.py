#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This script requires the nipy-data package to run. It is an example of
simultaneous motion correction and slice timing correction in multi-session fMRI
data from the FIAC 2005 dataset. Specifically, it uses the first two sessions of
subject 'fiac0'.

Usage:
  python space_time_realign.py

Two images will be created in the working directory for the realigned series::

    rarun1.nii
    rarun2.nii

Author: Alexis Roche, 2009.
"""
import os
from os.path import split as psplit, abspath

from nipy.algorithms.registration import FmriRealign4d

from nipy import load_image, save_image
from nipy.utils import example_data

# Input images are provided with the nipy-data package
runnames = [example_data.get_filename('fiac', 'fiac0', run + '.nii.gz')
            for run in ('run1', 'run2')]
runs = [load_image(run) for run in runnames]

# Declare interleaved ascending slice order
nslices = runs[0].shape[2]
slice_order = range(0, nslices, 2) + range(1, nslices, 2)
print('Slice order: %s' % slice_order)

# Spatio-temporal realigner
R = FmriRealign4d(runs, tr=2.5, slice_order=slice_order)

# Estimate motion within- and between-sessions
R.estimate(refscan=None)

# Resample data on a regular space+time lattice using 4d interpolation
# Save images
cwd = abspath(os.getcwd())
print('Saving results in: %s' % cwd)
for i in range(len(runs)):
    corr_run = R.resample(i)
    fname = 'ra' + psplit(runnames[i])[1]
    save_image(corr_run, fname)
