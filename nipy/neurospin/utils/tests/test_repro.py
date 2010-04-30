"""
Test the design_matrix utilities.

Note that the tests just looks whether the data produces has correct dimension,
not whether it is exact
"""

import numpy as np
import nipy.neurospin.utils.simul_2d_multisubject_fmri_dataset as simul
from nipy.neurospin.utils.reproducibility_measures import \
     voxel_reproducibility, cluster_reproducibility

from nipy.testing import (parametric, dec, assert_almost_equal,
                           assert_true)


def make_dataset(ampli_factor=1.0, nsubj=10):
    """
    Generate a standard multi-subject as a set of multi-subject 2D maps
    if null, no activation is added
    """
    nsubj = 10
    dimx = 60
    dimy = 60
    pos = 2*np.array([[ 6,  7], [10, 10], [15, 10]])
    ampli = ampli_factor*np.array([5, 6, 7])
    dataset = simul.make_surrogate_array(nbsubj=nsubj, dimx=dimx, dimy=dimy, 
                                         pos=pos, ampli=ampli, width=5.0, seed=1)
    return dataset


def apply_repro_analysis_analysis(dataset, thresholds=[3.0], method = 'crfx'):
    """
    perform the reproducibility  analysis according to the 
    """
    nsubj, dimx, dimy = dataset.shape
    
    func = np.reshape(dataset,(nsubj, dimx*dimy)).T
    var = np.ones((dimx*dimy, nsubj))
    xyz = np.reshape(np.indices((dimx, dimy,1)).T,(dimx*dimy,3))
    coord = xyz.astype(np.float)

    ngroups = 10
    sigma = 2.0
    csize = 10
    niter = 10
    verbose = 0
    swap = False

    kap = []
    clt = []
    for threshold in thresholds:
        kappa = []
        cls = []
        kwargs={'threshold':threshold,'csize':csize}        
        for i in range(niter):
            k = voxel_reproducibility(func, var, xyz, ngroups,
                                  method, swap, verbose, **kwargs)
            kappa.append(k)
            cld = cluster_reproducibility(func, var, xyz, ngroups, coord, sigma,
                                      method, swap, verbose, **kwargs)
            cls.append(cld)
        
        kap.append(np.array(kappa))
        clt.append(np.array(cls))
    kap = np.array(kap)
    clt = np.array(clt)
    return kap,clt


def test_repro1():
    # Test on the kappa values for a standard dataset using bootstrap
    dataset = make_dataset()
    kap,clt = apply_repro_analysis_analysis(dataset)
    assert_true((kap.mean()>0.3) & (kap.mean()<0.9))


def test_repro2():
    # Test on the cluster reproducibility values for a standard dataset
    # using cluster-level rfx, bootstrap
    dataset = make_dataset()
    kap,clt = apply_repro_analysis_analysis(dataset, thresholds=[5.0])
    assert_true(clt.mean()>0.5)

    
def test_repro3():
    # Test on the kappa values for a null dataset using cluster-level
    # rfx, bootstrap
    dataset = make_dataset(ampli_factor=0)
    kap,clt = apply_repro_analysis_analysis(dataset, thresholds=[5.0])
    assert_true(kap.mean(1)<0.3)


def test_repro4():
    # Test on the cluster repro. values for a null dataset using
    # cluster-level rfx, bootstrap
    dataset = make_dataset(ampli_factor=0)
    kap,clt = apply_repro_analysis_analysis(dataset, thresholds=[3.0])
    assert_true(clt.mean(1)<0.3)


def test_repro5():
    # Test on the kappa values for a non-null dataset using
    # cluster-level mfx, bootstrap
    dataset = make_dataset()
    kap,clt = apply_repro_analysis_analysis(dataset, method='cmfx')
    assert_true(kap.mean(1)>0.5)


def test_repro6():
    # Test on the kappa values for a non-null dataset using
    # cluster-level mfx, bootstrap
    dataset = make_dataset()
    kap,clt = apply_repro_analysis_analysis(dataset, method='cmfx')
    assert_true(clt.mean(1)>0.5)


def test_repro7():
    # Test on the kappa values for a standard dataset using jacknife
    # subsampling
    dataset = make_dataset(nsubj = 101)
    kap,clt = apply_repro_analysis_analysis(dataset, thresholds=[5.0])
    assert_true((kap.mean()>0.4))


def test_repro8():
    # Test on the kappa values for a standard dataset using jacknife
    # subsampling
    dataset = make_dataset(nsubj = 101)
    kap,clt = apply_repro_analysis_analysis(dataset, thresholds=[5.0])
    assert_true((clt.mean()>0.5))

