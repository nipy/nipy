# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Tests for bayesian_structural_analysis

Author : Bertrand Thirion, 2009
"""
#autoindent

import numpy as np
import scipy.stats as st
from nose.tools import assert_true

from nipy.testing import dec

import nipy.neurospin.graph.field as ff
import nipy.neurospin.utils.simul_multisubject_fmri_dataset as simul
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
import nipy.neurospin.spatial_models.structural_bfls as sbf
from nipy.neurospin.spatial_models.discrete_domain import domain_from_array


def make_bsa_2d(betas, theta=3., dmax=5., ths=0, thq=0.5, smin=0, 
                        nbeta=[0], method='simple'):
    """
    Function for performing bayesian structural analysis on a set of images.

    Fixme: 'quick' is not tested
    """
    ref_dim = np.shape(betas[0])
    nbsubj = betas.shape[0]
    xyz = np.array(np.where(betas[:1])).T
    nvox = np.size(xyz, 0)
    
    # create the field strcture that encodes image topology
    Fbeta = ff.Field(nvox)
    Fbeta.from_3d_grid(xyz.astype(np.int), 18)

    # get the functional information
    lbeta = np.array([np.ravel(betas[k]) for k in range(nbsubj)]).T

    # the voxel volume is 1.0
    g0 = 1.0/(1.0*nvox)
    bdensity = 1
    dom = domain_from_array(np.ones(ref_dim))

    if method=='simple':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_simple(dom, lbeta, dmax, thq, smin, ths,
                                       theta, g0, bdensity)    
    if method=='ipmi':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_ipmi(dom, lbeta, dmax, thq, smin, ths,
                                       theta, g0, bdensity)
    if method=='sbf':
        pval = 0.2
        group_map, AF, BF = sbf.Compute_Amers (
            dom, lbeta, dmax, theta, ths, pval)
    return AF, BF


@dec.slow    
def test_bsa_methods():
    # generate the data
    nbsubj=10
    dimx=60
    dimy=60
    pos = np.array([[12,  14],
                    [20, 20],
                    [40, 50]])
    # make a dataset with a nothing feature
    null_ampli = np.array([0, 0, 0])
    null_dataset = simul.surrogate_2d_dataset(nbsubj=nbsubj,
                                              dimx=dimx,
                                              dimy=dimy, 
                                              pos=pos,
                                              ampli=null_ampli,
                                              width=5.0,
                                              seed=1)
    null_betas = np.reshape(null_dataset, (nbsubj, dimx, dimy))
    # make a dataset with a something feature
    pos_ampli = np.array([5, 7, 6])
    pos_dataset = simul.surrogate_2d_dataset(nbsubj=nbsubj,
                                              dimx=dimx,
                                              dimy=dimy, 
                                              pos=pos,
                                              ampli=pos_ampli,
                                              width=5.0,
                                              seed=2)
    pos_betas = np.reshape(pos_dataset, (nbsubj, dimx, dimy))
    # set various parameters
    theta = float(st.t.isf(0.01, 100))
    dmax = 5./1.5
    half_subjs = nbsubj/2
    thq = 0.9
    smin = 5

    # tuple of tuples with each tuple being
    # (name_of_method, ths_value, data_set, test_function)
    algs_tests = (
        ('simple', half_subjs, null_betas, lambda AF, BF: AF.k == 0),
        ('ipmi', half_subjs, null_betas, lambda AF, BF: AF.k == 0),
        ('simple', 1, pos_betas, lambda AF, BF: AF.k>1),
        ('ipmi', 1, pos_betas, lambda AF, BF: AF.k>1),
        ('sbf', 1 , pos_betas, lambda AF, BF: AF.k>1))
    
    for name, ths, betas, test_func in algs_tests:
        # run the algo
        AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin, method = name)
        yield assert_true, test_func(AF, BF)
    

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
