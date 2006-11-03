#!/bin/env python
''' Overall mean

Compile a list of pre-existing effect size images for the first 4
subjects and the first run in the example dataset, and create an image
giving the mean of these effect size images across subjects.

Usage:

overall_mean.py

Creates overall_mean.img in working directory
'''

import numpy as N
import pylab

from neuroimaging.core.image.image import Image
from neuroimaging.ui.visualization.viewer import BoxViewer
from neuroimaging.utils.tests.data import repository

def subj_run_effect_img(subject=0, run=1):
    ''' Return effect image from example dataset for subject and run no 
    '''
    return Image(
        'FIAC/fiac%d/fonc%d/fsl/fmristat_run/contrasts/overall/effect.img'
        % (subject,run), repository)

def mean_overall(images):
    ''' Return mean over list of Images '''
    out = N.zeros(images[0].grid.shape, N.float64)
    for im in images:
        out += im[:]
    out /= len(images)
    return Image(out, grid=images[0].grid)

# Collect subject images for first run
subject_nos = range(4)
effect_imgs = [subj_run_effect_img(subject=i, run=1) for i in subject_nos]

# Create mean image and write to file
overall_mean_img = mean_overall(effect_imgs)
overall_mean_img.tofile('overall_mean.img', clobber=True)

# Show in orthogonal viewer
viewer = BoxViewer(overall_mean_img, colormap='spectral')
viewer.draw()
pylab.show()
