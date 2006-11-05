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
    for i in range(len(images)):
        out += images[i][:]
    out /= len(images)
    return Image(out, grid=images[0].grid)

SUBJECT_NOS = range(4)
EFFECT_IMGS = [subj_run_effect_img(subject=i) for i in SUBJECT_NOS]

OVERALL_MEAN = mean_overall(EFFECT_IMGS)
OVERALL_MEAN.tofile('overall_mean.img', clobber=True)

VIEWER = BoxViewer(OVERALL_MEAN, colormap='spectral')
VIEWER.draw()
pylab.show()
