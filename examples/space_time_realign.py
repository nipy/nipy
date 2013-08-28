#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This script requires the nipy-data package to run. It is an example of
simultaneous motion correction and slice timing correction in
multi-session fMRI data from the FIAC 2005 dataset. Specifically, it
uses the first two sessions of subject 'fiac0'.

Usage:
  python space_time_realign.py

Two images will be created in the working directory for the realigned series::

    rarun1.nii
    rarun2.nii

Author: Alexis Roche, 2009.
"""
from __future__ import print_function  # Python 2/3 compatibility

import os
from os.path import split as psplit, abspath
import numpy as np
from nipy.algorithms.registration import SpaceTimeRealign
from nipy import load_image, save_image
from nipy.utils import example_data

# Input images are provided with the nipy-data package
runnames = [example_data.get_filename('fiac', 'fiac0', run + '.nii.gz')
            for run in ('run1', 'run2')]
runs = [load_image(run) for run in runnames]

# Spatio-temporal realigner assuming interleaved ascending slice order
R = SpaceTimeRealign(runs, tr=2.5, slice_times='asc_alt_2', slice_info=2)

# If you are not sure what the above is doing, you can alternatively
# declare slice times explicitly using the following equivalent code
"""
tr = 2.5
nslices = runs[0].shape[2]
slice_times = (tr / float(nslices)) *\
    np.argsort(range(0, nslices, 2) + range(1, nslices, 2))
print('Slice times: %s' % slice_times)
R = SpaceTimeRealign(runs, tr=tr, slice_times=slice_times, slice_info=2)
"""

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
