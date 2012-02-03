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

from ...utils.simul_multisubject_fmri_dataset import surrogate_2d_dataset
from ..bayesian_structural_analysis import compute_BSA_simple
from ..discrete_domain import domain_from_binary_array


def make_bsa_2d(betas, theta=3., dmax=5., ths=0, thq=0.5, smin=0,
                        nbeta=[0], method='simple'):
    """
    Function for performing bayesian structural analysis on a set of images.

    Fixme: 'quick' is not tested
    """
    ref_dim = np.shape(betas[0])
    n_subj = betas.shape[0]
    xyz = np.array(np.where(betas[:1])).T
    nvox = np.size(xyz, 0)

    # get the functional information
    lbeta = np.array([np.ravel(betas[k]) for k in range(n_subj)]).T

    # the voxel volume is 1.0
    g0 = 1.0 / (1.0 * nvox)
    bdensity = 1
    dom = domain_from_binary_array(np.ones(ref_dim))

    if method == 'simple':
        group_map, AF, BF, likelihood = \
                   compute_BSA_simple(dom, lbeta, dmax, thq, smin, ths,
                                      theta, g0, bdensity)
    return AF, BF


@dec.slow
def test_bsa_methods():
    # generate the data
    n_subj = 5
    shape = (40, 40)
    pos = np.array([[12, 14],
                    [20, 20],
                    [30, 35]])
    # make a dataset with a nothing feature
    null_ampli = np.array([0, 0, 0])
    null_betas = surrogate_2d_dataset(n_subj=n_subj,
                                        shape=shape,
                                        pos=pos,
                                        ampli=null_ampli,
                                        width=5.0,
                                        seed=1)
    #null_betas = np.reshape(null_dataset, (n_subj, shape[0], shape[1]))
    # make a dataset with a something feature
    pos_ampli = np.array([5, 7, 6])
    pos_betas = surrogate_2d_dataset(n_subj=n_subj,
                                       shape=shape,
                                       pos=pos,
                                       ampli=pos_ampli,
                                       width=5.0,
                                       seed=2)
    #pos_betas = np.reshape(pos_dataset, (n_subj, shape[0], shape[1]))
    # set various parameters
    theta = float(st.t.isf(0.01, 100))
    dmax = 5. / 1.5
    half_subjs = n_subj / 2
    thq = 0.9
    smin = 5

    # tuple of tuples with each tuple being
    # (name_of_method, ths_value, data_set, test_function)
    algs_tests = (
        ('simple', half_subjs, null_betas, lambda AF, BF: AF.k == 0),
        ('simple', 1, pos_betas, lambda AF, BF: AF.k > 1))

    for name, ths, betas, test_func in algs_tests:
        # run the algo
        AF, BF = make_bsa_2d(betas, theta, dmax, ths, thq, smin, method=name)
        yield assert_true, test_func(AF, BF)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
