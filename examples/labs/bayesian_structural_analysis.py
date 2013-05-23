#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
This script generates a noisy multi-subject activation image dataset
and applies the Bayesian structural analysis on it

Requires matplotlib

Author : Bertrand Thirion, 2009-2013
"""
print(__doc__)

import numpy as np
import scipy.stats as st

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

import nipy.labs.utils.simul_multisubject_fmri_dataset as simul
from nipy.labs.spatial_models.bayesian_structural_analysis import\
    compute_landmarks
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_shape

    
def display_landmarks_2d(landmarks, hrois, stats):
    """ Plots the landmarks and associated rois as images"""
    shape = stats[0].shape
    n_subjects = len(stats)
    lmax = 0
    grp_map, density = np.zeros(shape), np.zeros(shape)
    if landmarks != None:
        domain = landmarks.domain
        grp_map = landmarks.map_label(domain.coord, .8, sigma).reshape(shape)
        density = landmarks.kernel_density(k=None, coord=domain.coord,
                                           sigma=sigma).reshape(shape)
        lmax = landmarks.k + 2
        
    # Figure 1: input data    
    fig_input = plt.figure(figsize=(8, 3.5))
    fig_input.text(.5,.9, "Input activation maps", ha='center')
    vmin, vmax = stats.min(), stats.max()
    for subject in range(n_subjects):
        plt.subplot(n_subjects / 5, 5, subject + 1)
        plt.imshow(stats[subject], interpolation='nearest',
                   vmin=vmin, vmax=vmax)
        plt.axis('off')

    # Figure 2: individual hrois
    fig_output = plt.figure(figsize=(8, 3.5))
    fig_output.text(.5, .9, "Individual landmark regions", ha="center")
    for subject in range(n_subjects):
        plt.subplot(n_subjects / 5, 5, subject + 1)
        lw = - np.ones(shape)
        if hrois[subject].k > 0:
            nls = hrois[subject].get_roi_feature('label')
            nls[nls == - 1] = np.size(landmarks) + 2
            for k in range(hrois[subject].k):
                np.ravel(lw)[hrois[subject].label == k] = nls[k]

        plt.imshow(lw, interpolation='nearest', vmin=-1, vmax=lmax)
        plt.axis('off')
        
    # Figure 3: Group-level results
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(grp_map, interpolation='nearest', vmin=-1, vmax=lmax)
    plt.title('group-level position 80% \n confidence regions', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=.8)

    plt.subplot(1, 2, 2)
    plt.imshow(density, interpolation='nearest')
    plt.title('Spatial density under h1', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=.8)


###############################################################################
# Main script
###############################################################################

# generate the data
n_subjects = 10
shape = (60, 60)
pos = np.array([[12, 14],
                [20, 20],
                [30, 20]])
ampli = np.array([5, 7, 6])
sjitter = 1.0
stats = simul.surrogate_2d_dataset(n_subj=n_subjects, shape=shape, pos=pos,
                                   ampli=ampli, width=5.0)

# set various parameters
threshold = float(st.t.isf(0.01, 100))
sigma = 4. / 1.5
prevalence_threshold = n_subjects * .25
prevalence_pval = 0.9
smin = 5
algorithm = 'co-occurrence' #  'density'

domain = grid_domain_from_shape(shape) 

# get the functional information
stats_ = np.array([np.ravel(stats[k]) for k in range(n_subjects)]).T
    
# run the algo
landmarks, hrois = compute_landmarks(
    domain, stats_, sigma, prevalence_pval, prevalence_threshold, 
    threshold, smin, method='prior', algorithm=algorithm)

display_landmarks_2d(landmarks, hrois, stats)
if landmarks is not None:
    landmarks.show()

plt.show()

