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
from nipy.labs.spatial_models.discrete_domain import domain_from_binary_array


def make_bsa_2d(betas, threshold=3., sigma=5., prevalence_threshold=0, 
                prevalence_pval=0.5, smin=0, method='simple', verbose=0):
    """
    Function for performing bayesian structural analysis
    on a set of images.

    Parameters
    ----------
    betas, array of shape (n_subjects, dimx, dimy) the data used
           Note that it is assumed to be a t- or z-variate
    threshold=3., float,
              first level threshold of betas
    sigma=5., float, expected between subject variability
    prevalence_threshold=0, float,
           null hypothesis for the prevalence statistic
    prevalence_pval=0.5, float,
             p-value of the null rejection
    smin=0, int,
            threshold on the nu_mber of contiguous voxels
            to make regions meaningful structures
    method= 'simple', string,
            estimation method used ; to be chosen among
            'simple', 'quick', 'loo', 'ipmi'
    verbose=0, verbosity mode

    Returns
    -------
    landmarks the landmark_regions instance describing the result
    hrois: list of hroi instances describing the individual data
    """
    ref_dim = np.shape(betas[0])
    n_subjects = betas.shape[0]

    # get the functional information
    stats = np.array([np.ravel(betas[k]) for k in range(n_subjects)]).T

    lmax = 0
    domain = domain_from_binary_array(np.ones(ref_dim))

    if method == 'simple':
        algorithm = 'standard'
    elif method == 'quick':
        algorithm = 'quick'
    else:
        raise ValueError('method is not correctly defined')
    
    landmarks, hrois = compute_landmarks(
        domain, stats, sigma, prevalence_pval, prevalence_threshold, 
        threshold, smin, method='prior', algorithm=algorithm)
    
    if landmarks != None:
        grp_map = landmarks.map_label(domain.coord, .8, sigma)
        grp_map.shape = ref_dim
        group_map = landmarks.map_label(domain.coord, 0.95,  sigma)
        density = landmarks.kernel_density(k=None, coord=domain.coord,
                                           sigma=sigma)
        group_map.shape = ref_dim

    if verbose == 0:
        return landmarks, hrois

    if landmarks != None:
        lmax = landmarks.k + 2
        landmarks.show()

    fig_output = plt.figure(figsize=(8, 3.5))
    fig_output.text(.5, .9, "Individual landmark regions", ha="center")
    for s in range(n_subjects):
        plt.subplot(n_subjects / 5, 5, s + 1)
        #ax.set_position([.02, .02, .96, .96])
        lw = - np.ones(ref_dim)
        if hrois[s] is not None:
            nls = hrois[s].get_roi_feature('label')
            nls[nls == - 1] = np.size(landmarks) + 2
            for k in range(hrois[s].k):
                np.ravel(lw)[hrois[s].label == k] = nls[k]

        plt.imshow(lw, interpolation='nearest', vmin=-1, vmax=lmax)
        plt.axis('off')

    fig_input = plt.figure(figsize=(8, 3.5))
    fig_input.text(.5,.9, "Input activation maps", ha='center')
    for s in range(n_subjects):
        plt.subplot(n_subjects / 5, 5, s + 1)
        plt.imshow(betas[s], interpolation='nearest', vmin=betas.min(),
                  vmax=betas.max())
        plt.axis('off')
        
    if landmarks is None:
        return landmarks, hrois
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(group_map, interpolation='nearest', vmin=-1, vmax=lmax)
    plt.title('Blob separation map', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=.8)

    plt.subplot(1, 3, 2)
    plt.imshow(grp_map, interpolation='nearest', vmin=-1, vmax=lmax)
    plt.title('group-level position 80% \n confidence regions', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=.8)

    plt.subplot(1, 3, 3)
    density.shape = ref_dim
    plt.imshow(density, interpolation='nearest')
    plt.title('Spatial density under h1', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=.8)

    return landmarks, hrois


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
betas = simul.surrogate_2d_dataset(n_subj=n_subjects, shape=shape, pos=pos,
                                   ampli=ampli, width=5.0)

# set various parameters
threshold = float(st.t.isf(0.01, 100))
sigma = 4. / 1.5
prevalence_threshold = n_subjects * .25
prevalence_pval = 0.9
verbose = 1
smin = 5
method = 'quick'  # 'simple' #

# run the algo
landmarks, hrois = make_bsa_2d(betas, threshold, sigma, prevalence_threshold, 
                               prevalence_pval, smin, method, verbose=verbose)
plt.show()
