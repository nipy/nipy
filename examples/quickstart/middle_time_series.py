#!/bin/env python
'''
Plot the mean intensity of a 10x10x10 voxel block from the middle of
the first functional run from the first subject in the example
dataset

Usage
middle_time_series.py
'''

import numpy as N
import pylab

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils.tests.data import repository

subject_no=0
run_no = 1
func_img = fMRIImage('FIAC/fiac%d/fonc%d/fsl/filtered_func_data.img' %
                     (subject_no, run_no),
                     datasource=repository)

# Create array of same shape as one time point of functional image
mask_arr = N.zeros(func_img.grid.shape[1:])

# Set middle block of mask_arr to 1 (block size is offset*2)
offset=5
middle_block_def = [slice(i/2-offset, i/2+offset, 1) for i in mask_arr.shape]
mask_arr[middle_block_def] = 1

# Number of non-zero voxels set in mask array
nvox = mask_arr.sum()

# Take mean of voxels for each time point within mask
n_tps = func_img.grid.shape[0]
tp_means = N.zeros(n_tps)
for i in range(n_tps):
    tp_means[i] = (mask_arr * func_img[slice(i, i+1)]).sum() / nvox

pylab.plot(tp_means)
pylab.show()
