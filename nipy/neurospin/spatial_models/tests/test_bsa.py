"""
Tests for bayesian_structural_analysis

Author : Bertrand Thirion, 2009
"""
#autoindent

import numpy as np

import scipy.stats as st

import nipy.neurospin.graph.field as ff
import nipy.neurospin.utils.simul_multisubject_fmri_dataset as simul
import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
import nipy.neurospin.spatial_models.structural_bfls as sbf

from nipy.testing import assert_true, dec


def make_bsa_2d(betas, theta=3., dmax=5., ths=0, thq=0.5, smin=0, 
                        nbeta=[0],method='simple'):
    """ Function for performing bayesian structural analysis on a set of images.
    """
    ref_dim = np.shape(betas[0])
    nbsubj = betas.shape[0]
    xyz = np.array(np.where(betas[:1])).T
    nvox = np.size(xyz, 0)
    
    # create the field strcture that encodes image topology
    Fbeta = ff.Field(nvox)
    Fbeta.from_3d_grid(xyz.astype(np.int), 18)

    # Get  coordinates in mm
    tal = xyz.astype(np.float)

    # get the functional information
    lbeta = np.array([np.ravel(betas[k]) for k in range(nbsubj)]).T

    # the voxel volume is 1.0
    g0 = 1.0/(1.0*nvox)
    bdensity = 1
    affine = np.eye(4)
    shape = (1,ref_dim[0],ref_dim[1])
    
    if method=='ipmi':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_ipmi(Fbeta, lbeta, tal, dmax,xyz, affine, 
                                               shape, thq,
                                        smin, ths, theta, g0, bdensity)
    if method=='simple':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_simple(Fbeta, lbeta, tal, dmax,xyz,
                                          affine, shape, thq, smin, ths, 
                                          theta, g0)
    if method=='dev':
        group_map, AF, BF, likelihood = \
                   bsa.compute_BSA_dev(Fbeta, lbeta, tal, dmax, xyz, affine, 
                                              shape, thq,
                                       smin, ths, theta, g0, bdensity)
    if method=='sbf':
        pval = 0.2
        group_map, AF, BF = sbf.Compute_Amers (Fbeta, lbeta, xyz, affine, 
                                                      shape,
                                               tal, dmax, theta, ths ,pval)
    return AF, BF


@dec.slow    
def test_bsa_methods():
    # generate the data
    nbsubj=10
    dimx=60
    dimy=60
    pos = 2*np.array([[ 6,  7],
                      [10, 10],
                      [15, 10]])
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
    algs_tests = (('simple', half_subjs, null_betas, lambda AF, BF: AF == None),
                  ('dev', half_subjs, null_betas, lambda AF, BF: AF == None),
                  ('ipmi', half_subjs, null_betas, lambda AF, BF: AF == None),
                  ('simple', 1, pos_betas, lambda AF, BF: AF.k>1),
                  ('dev', 1, pos_betas, lambda AF, BF: AF.k>1),
                  ('ipmi', 1, pos_betas, lambda AF, BF: AF.k>1),
                  )
    for name, ths, betas, test_func in algs_tests:
        # run the algo
        AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin, method = name)
        yield assert_true, test_func(AF, BF)
        

