# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the design_matrix utilities.

Note that the tests just looks whether the data produces has correct dimension,
not whether it is exact
"""
from __future__ import absolute_import

import numpy as np
from ..simul_multisubject_fmri_dataset import surrogate_2d_dataset
from ..reproducibility_measures import (voxel_reproducibility, 
                                        cluster_reproducibility,
                                        peak_reproducibility)

def make_dataset(ampli_factor=1.0, n_subj=10):
    """
    Generate a standard multi-subject as a set of multi-subject 2D maps
    if null, no activation is added
    """
    n_subj = 10
    shape = (40, 40)
    pos = 2 * np.array([[ 6,  7], [10, 10], [15, 10]])
    ampli = ampli_factor * np.array([5, 6, 7])
    dataset = surrogate_2d_dataset(n_subj=n_subj, shape=shape, pos=pos, 
                                   ampli=ampli, width=5.0, seed=1)
    return dataset


def apply_repro_analysis(dataset, thresholds=[3.0], method = 'crfx'):
    """
    perform the reproducibility  analysis according to the 
    """
    from nipy.labs.spatial_models.discrete_domain import \
        grid_domain_from_binary_array

    n_subj, dimx, dimy = dataset.shape
    
    func = np.reshape(dataset,(n_subj, dimx * dimy)).T
    var = np.ones((dimx * dimy, n_subj))
    domain = grid_domain_from_binary_array(np.ones((dimx, dimy, 1)))

    ngroups = 5
    sigma = 2.0
    csize = 10
    niter = 5
    verbose = 0
    swap = False

    kap, clt, pkd = [], [], []
    for threshold in thresholds:
        kappa, cls, pks = [], [], []
        kwargs = {'threshold':threshold, 'csize':csize}        
        for i in range(niter):
            k = voxel_reproducibility(func, var, domain, ngroups,
                                  method, swap, verbose, **kwargs)
            kappa.append(k)
            cld = cluster_reproducibility(func, var, domain, ngroups, sigma,
                                      method, swap, verbose, **kwargs)
            cls.append(cld)
            pk = peak_reproducibility(func, var, domain, ngroups, sigma,
                                      method, swap, verbose, **kwargs)
            pks.append(pk)
        
        kap.append(np.array(kappa))
        clt.append(np.array(cls))
        pkd.append(np.array(pks))
    kap = np.array(kap)
    clt = np.array(clt)
    pkd = np.array(pkd)
    return kap, clt, pkd

def test_repro1():
    """
    Test on the kappa values for a standard dataset
    using bootstrap
    """
    dataset = make_dataset()
    kap, clt, pks = apply_repro_analysis(dataset)
    assert ((kap.mean() > 0.3) & (kap.mean() < 0.9))
    assert (pks.mean() > 0.4)

def test_repro2():
    """
    Test on the cluster reproducibility values for a standard dataset
    using cluster-level rfx, bootstrap
    """
    dataset = make_dataset()
    kap, clt, pks = apply_repro_analysis(dataset, thresholds=[5.0])
    assert (clt.mean()>0.5)

    
def test_repro3():
    """
    Test on the kappa values for a null dataset
    using cluster-level rfx, bootstrap
    """
    dataset = make_dataset(ampli_factor=0)
    kap, clt, pks = apply_repro_analysis(dataset, thresholds=[4.0])
    assert (kap.mean(1) < 0.3)
    assert (clt.mean(1) < 0.3)

def test_repro5():
    """
    Test on the kappa values for a non-null dataset
    using cluster-level mfx, bootstrap
    """
    dataset = make_dataset()
    kap, clt, pks = apply_repro_analysis(dataset, method='cmfx')
    assert (kap.mean(1) > 0.5)
    assert (clt.mean(1) > 0.5)

def test_repro7():
    """
    Test on the kappa values for a standard dataset
    using jacknife subsampling
    """
    dataset = make_dataset(n_subj = 101)
    kap, clt, pks = apply_repro_analysis(dataset, thresholds=[5.0])
    assert ((kap.mean() > 0.4))
    assert ((clt.mean() > 0.5))    

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])


