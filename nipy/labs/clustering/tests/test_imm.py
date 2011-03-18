"""
Test the Infinite GMM.

Author : Bertrand Thirion, 2010 
"""

#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

import numpy as np
from ..imm import IMM, MixedIMM, co_labelling

def test_colabel():
    """
    test the co_lavbelling functionality
    """
    z = np.array([0,1,1,0,2])
    c = co_labelling(z).todense()
    tc = np.array([[ 1.,  0.,  0.,  1.,  0.],
                   [ 0.,  1.,  1.,  0.,  0.],
                   [ 0.,  1.,  1.,  0.,  0.],
                   [ 1.,  0.,  0.,  1.,  0.],
                   [ 0.,  0.,  0.,  0.,  1.]])
    assert (c==tc).all()

def test_imm_loglike_1D():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood
    """
    n = 100
    dim = 1
    alpha = .5
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100)

    # sampling
    like =  igmm.sample(x, niter=300)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.25*dim

def test_imm_loglike_known_groups():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood
    """
    n = 50
    dim = 1
    alpha = .5
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)
    kfold = np.floor(np.random.rand(n)*5).astype(np.int)
    
    # warming
    igmm.sample(x, niter=100)

    # sampling
    like =  igmm.sample(x, niter=300, kfold=kfold)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.25*dim

def test_imm_loglike_1D_k10():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood

    Here k-fold cross validation is used(k=10)
    """
    n = 50
    dim = 1
    alpha = .5
    k = 5
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100, kfold=k)

    # sampling
    like =  igmm.sample(x, niter=300, kfold=k)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.25*dim


def test_imm_loglike_2D_fast():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood

    fast version
    """
    n = 100
    dim = 2
    alpha = .5
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100, init=True)

    # sampling
    like =  igmm.sample(x, niter=300)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.25*dim

def test_imm_loglike_2D():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood

    slow (cross-validated) but accurate.
    """
    n = 50
    dim = 2
    alpha = .5
    k = 5
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100, init=True, kfold=k)

    # sampling
    like =  igmm.sample(x, niter=300, kfold=k)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.25*dim


def test_imm_loglike_2D_a0_1():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood

    the difference is that now alpha=.1
    """
    n = 100
    dim = 2
    alpha = .1
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100, init=True)

    # sampling
    like =  igmm.sample(x, niter=300)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.2*dim

def test_imm_wnc():
    """
    Test the basic imm_wnc  
    """
    n = 50
    dim = 1
    alpha = .5
    g0 = 1.
    x = np.random.rand(n, dim)
    x[:.3*n] *= .2
    x[:.1*n] *= .3

    # instantiate
    migmm = MixedIMM(alpha, dim)
    migmm.set_priors(x)
    migmm.set_constant_densities(null_dens=g0)
    ncp = 0.5*np.ones(n)
    
    # warming
    migmm.sample(x, null_class_proba=ncp, niter=100, init=True)
    g = np.reshape(np.linspace(0, 1, 101), (101, dim))

    #sampling
    like, pproba =  migmm.sample(x, null_class_proba=ncp, niter=300,
                         sampling_points=g)

    # the density should sum to 1
    ds = 0.01*like.sum()
    assert ds<1
    assert ds>.8
    assert np.sum(pproba>.5)>1
    assert np.sum(pproba<.5)>1

def test_imm_wnc1():
    """
    Test the basic imm_wnc, where the probaility under the null are random  
    """
    n = 50
    dim = 1
    alpha = .5
    g0 = 1.
    x = np.random.rand(n, dim)
    x[:.3*n] *= .2
    x[:.1*n] *= .3

    # instantiate
    migmm = MixedIMM(alpha, dim)
    migmm.set_priors(x)
    migmm.set_constant_densities(null_dens=g0)
    ncp = np.random.rand(n)
    
    # warming
    migmm.sample(x, null_class_proba=ncp, niter=100, init=True)
    g = np.reshape(np.linspace(0, 1, 101), (101, dim))

    #sampling
    like, pproba =  migmm.sample(x, null_class_proba=ncp, niter=300,
                         sampling_points=g)

    # the density should sum to 1
    ds = 0.01*like.sum()
    assert ds<1
    assert ds>.8
    assert np.sum(pproba>.5)>1
    assert np.sum(pproba<.5)>1


def test_imm_wnc2():
    """
    Test the basic imm_wnc when null class is shrunk to 0
    """
    n = 50
    dim = 1
    alpha = .5
    g0 = 1.
    x = np.random.rand(n, dim)
    x[:.3*n] *= .2
    x[:.1*n] *= .3

    # instantiate
    migmm = MixedIMM(alpha, dim)
    migmm.set_priors(x)
    migmm.set_constant_densities(null_dens=g0)
    ncp = np.zeros(n)
    
    # warming
    migmm.sample(x, null_class_proba=ncp, niter=100, init=True)

    #sampling
    like, pproba =  migmm.sample(x, null_class_proba=ncp, niter=300)
    assert like.min()>.1
    assert like.max()<5.
    assert (pproba==ncp).all()

def test_imm_wnc3():
    """
    Test the basic imm_wnc when null class is of proba 1 (nothing is estimated)
    """
    n = 50
    dim = 1
    alpha = .5
    g0 = 1.
    x = np.random.rand(n, dim)
    x[:.3*n] *= .2
    x[:.1*n] *= .3

    # instantiate
    migmm = MixedIMM(alpha, dim)
    migmm.set_priors(x)
    migmm.set_constant_densities(null_dens=g0)
    ncp = np.ones(n)
    
    # warming
    migmm.sample(x, null_class_proba=ncp, niter=100, init=True)

    #sampling
    like, pproba =  migmm.sample(x, null_class_proba=ncp, niter=300)
    assert (pproba==ncp).all()
    

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


