# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Several basic tests
for hierarchical clustering procedures
should be cast soon in a nicer unitest framework

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
from numpy.random import randn
from nipy.neurospin.eda.dimension_reduction import CCA, MDS, knn_Isomap, \
        eps_Isomap, infer_latent_dim, PCA

def test_cca():
    """
    Basic (valid) test of the CCA
    """
    X = randn(100,3)
    Y = randn(100,3)
    cc1 = CCA(X,Y)
    A = randn(3,3)
    Z = np.dot(X,A)
    cc2 = CCA(X,Z)
    cc3 = CCA(Y,Z)
    test = (np.sum((cc1-cc3)**2)<1.e-7)&(np.min(cc2>1.e-7))
    assert test

def test_mds():
    """
    Test of the multi-dimensional scaling algorithm
    """
    X = randn(10,3)
    M = MDS(X, rdim=2)
    u = M.train()
    x = X[:2]
    a = M.test(x)
    eps = 1.e-12
    test = np.sum(a-u[:2])**2<eps
    assert test

def test_pca():
    """
    Test of the multi-dimensional scaling algorithm
    """
    X = randn(10,3)
    P = PCA(X, rdim=2)
    u = P.train()
    x = X[:2]
    a = P.test(x)
    eps = 1.e-12
    test = np.sum(a-u[:2])**2<eps
    print a, u[:2]
    assert test

def test_pca_and_mds():
    """
    """
    X = randn(10,3)
    M = MDS(X, rdim=2)
    u1 = M.train()
    P = PCA(X,rdim=2)
    u2 = P.train()
    eta = np.dot(u1.T,u2)
    delta = eta - np.diag(np.diag(eta))
    print delta, eta
    assert (np.sum(delta**2)<1.e-12*np.sum(eta**2))
    
    

def test_knn_isomap():
    """
    Test of the isomap algorithm on knn graphs
    this one fails if the graph as more than one connected component
    to avoid this we use a non-random example
    """
    prng = np.random.RandomState(seed=2)
    X = prng.randn(10,3)
    M = knn_Isomap(X, rdim=1)
    u = M.train(k=2)
    x = X[:2,:]
    a = M.test(x)
    eps = 1.e-12
    test = np.sum(a-u[:2,:])**2<eps
    assert test
    
def test_eps_isomap():
    """
    Test of the esp_isomap procedure
    this one fails if the graph as more than one connected component
    to avoid this we use a non-random example
    """
    prng = np.random.RandomState(seed=2)
    X = prng.randn(10,3)
    M = eps_Isomap(X, rdim=2)
    u = M.train(eps = 2.)
    x = X[:2,:]
    a = M.test(x)
    eps = 1.e-12
    test = np.sum((a-u[:2])**2)<eps
    assert test

def test_dimension_estimation():
    """
    Test the dimension selection procedures
    """
    from numpy.random import randn
    k = 2
    x = 10*np.dot(np.dot(randn(100,k),np.eye(k)),randn(k,10))
    x += randn(100,10)
    ek = infer_latent_dim(x)
    print k, ek
    assert(k==ek)

def test_dimension_estimation_2():
    """
    Test the dimension selection procedures
    """
    from numpy.random import randn
    k = 5
    x = 10*np.dot(np.dot(randn(100,k),np.eye(k)),randn(k,10))
    x += randn(100,10)
    ek = infer_latent_dim(x)
    print k, ek
    assert(k==ek)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
