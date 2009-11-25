"""
Several basic tests
for hierarchical clustering procedures
should be cast soon in a nicer unitest framework

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
from numpy.random import randn
from nipy.neurospin.eda.dimension_reduction import CCA, MDS, knn_Isomap, \
        eps_Isomap

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


def test_knn_isomap():
    """
    Test of the isomap algorithm on knn graphs
    """
    X = randn(10,3)
    M = knn_Isomap(X, rdim=1)
    u = M.train(k=2)
    x = X[:2,:]
    a = M.test(x)
    eps = 1.e12
    test = np.sum(a-u[:2,:])**2<eps
    print np.sum(a-u[:2,:])**2
    assert test
    
def test_eps_isomap():
    """
    Test of the esp_isomap procedure
    this one sometimes fails...not sure why
    """
    X = randn(10,3)
    M = eps_Isomap(X, rdim=1)
    u = M.train(eps = 2.)
    x = X[:2,:]
    a = M.test(x)
    eps = 1.e12
    test = np.sum((a-u[:2])**2)<eps
    print np.sum((a-u[:2])**2), a-u[:2]
    assert test


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
