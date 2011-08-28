#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

import numpy as np
from nose.tools import assert_true
seed = 1
nr = np.random.RandomState([seed])
from ..gmm import GMM, best_fitting_GMM


def test_em_loglike0():
    """ Test that the likelihood of the GMM is expected on standard data
    """
    dim = 1
    k = 1
    n = 1000
    x = nr.randn(n,dim)
    lgmm = GMM(k, dim)
    lgmm.initialize(x)
    lgmm.estimate(x)
    ll = lgmm.average_log_like(x)
    ent = 0.5 * (1 + np.log(2 * np.pi))
    assert_true(np.absolute(ll + ent) < 3. / np.sqrt(n))

def test_em_loglike1():
    """Test that the likelihood of the GMM is expected on standard data
    """
    dim = 1
    k = 3
    n = 1000
    x = nr.randn(n, dim)
    lgmm = GMM(k, dim)
    lgmm.initialize(x)
    lgmm.estimate(x)
    ll = lgmm.average_log_like(x)
    ent = 0.5 * (1 + np.log(2 * np.pi))
    assert_true(np.absolute(ll + ent) < 3. / np.sqrt(n))

def test_em_loglike2():
    """ Test that the likelihood of the GMM is expected on standard data
    """
    dim = 1
    k = 1
    n = 1000
    scale = 3.
    offset = 4.
    x = offset + scale * nr.randn(n, dim)
    lgmm = GMM(k, dim)
    lgmm.initialize(x)
    lgmm.estimate(x)
    ll = lgmm.average_log_like(x)
    ent = 0.5 * (1 + np.log(2 * np.pi * scale ** 2))
    assert_true(np.absolute(ll + ent) < 3. / np.sqrt(n))

def test_em_loglike3():
    """ Test that the likelihood of the GMM is expected on standard data
    """
    dim = 2
    k = 1
    n = 1000
    scale = 3.
    offset = 4.
    x = offset + scale * nr.randn(n,dim)
    lgmm = GMM(k, dim)
    lgmm.initialize(x)
    lgmm.estimate(x)
    ll = lgmm.average_log_like(x)
    ent = dim * 0.5 * (1 + np.log(2 * np.pi * scale ** 2))
    assert_true(np.absolute(ll + ent) < dim * 3. / np.sqrt(n))

def test_em_loglike4():
    """Test that the likelihood of the GMM is expected on standard data
    """
    dim = 5
    k = 1
    n = 1000
    scale = 3.
    offset = 4.
    x = offset + scale * nr.randn(n, dim)
    lgmm = GMM(k,dim)
    lgmm.initialize(x)
    lgmm.estimate(x)
    ll = lgmm.average_log_like(x)
    ent = dim * 0.5 * (1 + np.log(2 * np.pi * scale ** 2))
    assert_true(np.absolute(ll + ent) < dim * 3. / np.sqrt(n))

def test_em_loglike5():
    """Test that the likelihood of the GMM is expected on standard data
    """
    dim = 2
    k = 1
    n = 1000
    scale = 3.
    offset = 4.
    x = offset + scale * nr.randn(n, dim)
    y = offset + scale * nr.randn(n, dim)
    lgmm = GMM(k, dim)
    lgmm.initialize(x)
    lgmm.estimate(x)
    ll = lgmm.average_log_like(y)
    ent = dim * 0.5 * (1 + np.log(2 * np.pi * scale ** 2))
    assert_true(np.absolute(ll + ent) < dim * 3. / np.sqrt(n))

def test_em_loglike6():
    """ Test that the likelihood of shifted data is lower 
    than the likelihood of non-shifted data
    """
    dim = 1
    k = 1
    n = 100
    offset = 3.
    x = nr.randn(n, dim)
    y = offset + nr.randn(n, dim)
    lgmm = GMM(k, dim)
    lgmm.initialize(x)
    lgmm.estimate(x)
    ll1 =  lgmm.average_log_like(x)
    ll2 = lgmm.average_log_like(y)
    assert_true( ll2 < ll1)

def test_em_selection():
    """
    test that the basic GMM-based model selection tool
    returns something sensible
    (i.e. the gmm used to represent the data has indeed one or two classes)
    """
    dim = 2
    x = np.concatenate((nr.randn(100, dim), 3 + 2 * nr.randn(100, dim)))

    krange = range(1, 10)
    lgmm = best_fitting_GMM(x, krange, prec_type='full',
                            niter=100, delta = 1.e-4, ninit=1)
    assert_true(lgmm.k < 4)
    

def test_em_gmm_full():
    """
    Computing the BIC value for different configurations
    of a GMM with ful diagonal matrices
    The BIC should be maximal for a number of classes of 1  or 2
    """
    # generate some data
    dim = 2
    x = np.concatenate((nr.randn(100, dim), 3 + 2 * nr.randn(100, dim)))
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-4

    bic = np.zeros(5)
    for k in range(1,6):
        lgmm = GMM(k, dim)
        lgmm.initialize(x)
        bic[k - 1] = lgmm.estimate(x, maxiter, delta)

    assert_true(bic[4] < bic[1])


def test_em_gmm_diag():
    """
    Computing the BIC value for GMMs with different number of classes,
    with diagonal covariance models
    The BIC should maximal for a number of classes of 1  or 2
    """
    # generate some data
    dim = 2
    x = np.concatenate((nr.randn(1000, dim), 3 + 2 * nr.randn(1000, dim)))
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-8
    prec_type = 'diag'

    bic = np.zeros(5)
    for k in range(1, 6):
        lgmm = GMM(k, dim, prec_type)
        lgmm.initialize(x)
        bic[k - 1] = lgmm.estimate(x, maxiter, delta)

    z = lgmm.map_label(x)
    assert_true(z.max() + 1 == lgmm.k)
    assert_true(bic[4] < bic[1])


def test_em_gmm_multi():
    """Playing with various initilizations on the same data
    """
    # generate some data
    dim = 2
    x = np.concatenate((nr.randn(1000, dim), 3 + 2 * nr.randn(100, dim)))
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-4
    ninit = 5
    k = 2
    
    lgmm = GMM(k,dim)
    bgmm = lgmm.initialize_and_estimate(x, niter=maxiter, delta=delta, 
                                        ninit=ninit)
    bic = bgmm.evidence(x)
    
    assert_true(np.isfinite(bic))
    
def test_em_gmm_largedim():
    """testing the GMM model in larger dimensions
    """
    # generate some data
    dim = 10
    x = nr.randn(100, dim)
    x[:30] += 2
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-4
    
    for k in range(2, 3):
        lgmm = GMM(k,dim)
        bgmm = lgmm.initialize_and_estimate(x, None, maxiter, delta, ninit=5)
        
    z = bgmm.map_label(x)

    # define the correct labelling
    u = np.zeros(100)
    u[:30] = 1

    #check the correlation between the true labelling
    # and the computed one
    eta = np.absolute(np.dot(z - z.mean(), u - u.mean()) /\
                          (np.std(z) * np.std(u) * 100))
    assert_true(eta > 0.3)

def test_em_gmm_heterosc():
    """
    testing the model in very ellipsoidal data:
    compute the bic values for several values of k 
    and check that the macimal is 1 or 2
    """
    # generate some data
    dim = 2
    x = nr.randn(100, dim)
    x[:50] += 3
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-4
    
    bic = np.zeros(5)
    for k in range(1,6):
        lgmm = GMM(k, dim)
        lgmm.initialize(x)
        bic[k - 1] = lgmm.estimate(x, maxiter, delta, 0)

    assert_true(bic[4] < bic[1])

        
def test_em_gmm_cv():
    """Comparison of different GMMs using cross-validation
    """
    # generate some data
    dim = 2
    xtrain = np.concatenate((nr.randn(100, dim), 3 + 2 * nr.randn(100, dim)))
    xtest = np.concatenate((nr.randn(1000, dim), 3 + 2 * nr.randn(1000, dim)))
    
    #estimate different GMMs for xtrain, and test it on xtest
    prec_type = 'full'
    k = 2        
    maxiter = 300
    delta = 1.e-4
    ll = []
    
    # model 1
    lgmm = GMM(k,dim,prec_type)
    lgmm.initialize(xtrain)
    bic = lgmm.estimate(xtrain,maxiter, delta)
    ll.append(lgmm.test(xtest).mean())

    # model 2
    prec_type='diag'
    lgmm = GMM(k, dim, prec_type)
    lgmm.initialize(xtrain)
    bic = lgmm.estimate(xtrain, maxiter, delta)
    ll.append(lgmm.test(xtest).mean())
        
    for  k in [1, 3, 10]:
        lgmm = GMM(k,dim,prec_type)
        lgmm.initialize(xtrain)
        ll.append(lgmm.test(xtest).mean())
            
    assert_true(ll[4]<ll[1])


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
