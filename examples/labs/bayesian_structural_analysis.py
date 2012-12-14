#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
This script generates a noisy multi-subject activation image dataset
and applies the Bayesian structural analysis on it

Requires matplotlib

Author : Bertrand Thirion, 2009-2011
"""
print(__doc__)

import numpy as np
import scipy.stats as st

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

import nipy.labs.utils.simul_multisubject_fmri_dataset as simul
import nipy.labs.spatial_models.bayesian_structural_analysis as bsa
from nipy.labs.spatial_models.discrete_domain import domain_from_binary_array


def make_bsa_2d(betas, theta=3., dmax=5., ths=0, thq=0.5, smin=0,
                method='simple', verbose=0):
    """
    Function for performing bayesian structural analysis
    on a set of images.

    Parameters
    ----------
    betas, array of shape (nsubj, dimx, dimy) the data used
           Note that it is assumed to be a t- or z-variate
    theta=3., float,
              first level threshold of betas
    dmax=5., float, expected between subject variability
    ths=0, float,
           null hypothesis for the prevalence statistic
    thq=0.5, float,
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
    AF the landmark_regions instance describing the result
    BF: list of hroi instances describing the individual data
    """
    ref_dim = np.shape(betas[0])
    nsubj = betas.shape[0]
    xyz = np.array(np.where(betas[:1])).T.astype(np.int)

    # Get  coordinates in mm
    xyz = xyz[:, 1:] # switch to dimension 2
    coord = xyz.astype(np.float)

    # get the functional information
    lbeta = np.array([np.ravel(betas[k]) for k in range(nsubj)]).T

    lmax = 0
    bdensity = 1
    dom = domain_from_binary_array(np.ones(ref_dim))

    if method == 'simple':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_simple(dom, lbeta, dmax, thq, smin, ths,
                                          theta)
    if method == 'quick':
        likelihood = np.zeros(ref_dim)
        group_map, AF, BF, coclustering = \
                   bsa.compute_BSA_quick(dom, lbeta, dmax, thq, smin, ths,
                                         theta)
    if method == 'loo':
        mll, ll0 = bsa.compute_BSA_loo(dom, lbeta, dmax, thq, smin, ths,
                                          theta, bdensity)
        return mll, ll0

    if method not in['loo', 'simple', 'quick']:
        raise ValueError('method is not correctly defined')

    if verbose == 0:
        return AF, BF

    if AF != None:
        lmax = AF.k + 2
        AF.show()

    group_map.shape = ref_dim
    plt.figure(figsize=(8, 3))
    ax = plt.subplot(1, 3, 1)
    plt.imshow(group_map, interpolation='nearest', vmin=-1, vmax=lmax)
    plt.title('Blob separation map', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=.8)

    if AF != None:
        group_map = AF.map_label(coord, 0.95, dmax)
        group_map.shape = ref_dim

    plt.subplot(1, 3, 2)
    plt.imshow(group_map, interpolation='nearest', vmin=-1, vmax=lmax)
    plt.title('group-level position 95% \n confidence regions', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=.8)

    plt.subplot(1, 3, 3)
    likelihood.shape = ref_dim
    plt.imshow(likelihood, interpolation='nearest')
    plt.title('Spatial density under h1', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=.8)


    fig_output = plt.figure(figsize=(8, 3.5))
    fig_output.text(.5, .9, "Individual landmark regions", ha="center")
    for s in range(nsubj):
        ax = plt.subplot(nsubj / 5, 5, s + 1)
        #ax.set_position([.02, .02, .96, .96])
        lw = - np.ones(ref_dim)
        if BF[s] is not None:
            nls = BF[s].get_roi_feature('label')
            nls[nls == - 1] = np.size(AF) + 2
            for k in range(BF[s].k):
                np.ravel(lw)[BF[s].label == k] = nls[k]

        plt.imshow(lw, interpolation='nearest', vmin=-1, vmax=lmax)
        plt.axis('off')

    fig_input = plt.figure(figsize=(8, 3.5))
    fig_input.text(.5,.9, "Input activation maps", ha='center')
    for s in range(nsubj):
        plt.subplot(nsubj / 5, 5, s + 1)
        plt.imshow(betas[s], interpolation='nearest', vmin=betas.min(),
                  vmax=betas.max())
        plt.axis('off')
    return AF, BF


###############################################################################
# Main script
###############################################################################

# generate the data
n_subj = 10
shape = (60, 60)
pos = np.array([[12, 14],
                [20, 20],
                [30, 20]])
ampli = np.array([5, 7, 6])
sjitter = 1.0
betas = simul.surrogate_2d_dataset(n_subj=n_subj, shape=shape, pos=pos,
                                   ampli=ampli, width=5.0)

# set various parameters
theta = float(st.t.isf(0.01, 100))
dmax = 4. / 1.5
ths = n_subj / 4
thq = 0.9
verbose = 1
smin = 5
method = 'simple' # 'quick' #  'loo' #

# run the algo
AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin, method,
                     verbose=verbose)
plt.show()
