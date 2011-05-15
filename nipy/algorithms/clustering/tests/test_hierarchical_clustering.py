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
from ..hierarchical_clustering import (average_link_graph, 
                                       average_link_graph_segment,
                                       ward, ward_quick, 
                                       ward_segment, ward_field_segment,
                                       ward_quick_segment)
from nipy.algorithms.graph.graph import WeightedGraph, knn
from nipy.algorithms.graph.field import field_from_graph_and_data

def alg_test_basic(n=100,k=5):
    """
    Check that we obtain the correct solution in a simplistic case 
    """
    np.random.seed(0)
    x = np.random.randn(n, 2)
    x[:int(0.7*n)] += 3
    G = knn(x, k)
    t = average_link_graph(G)
    u = t.split(2)
    v = np.zeros(n)
    v[:int(0.7*n)]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def alg_test_2():
    """
    Check that we hasnle the case of a graph with too many
    connected components
    """
    np.random.seed(0)
    n = 100
    k = 5
    x = np.random.randn(n, 2)
    x[:int(0.3*n)] += 10
    x[int(0.8*n):] -= 10
    G = knn(x, k)
    t = average_link_graph(G)
    u = t.split(2)
    assert(u.max()==2)

def alg_test_3(n=100,k=5):
    """
    Check that we obtain the correct solution in a simplistic case 
    """
    np.random.seed(0)
    x = np.random.randn(n, 2)
    x[:int(0.7*n)] += 3
    G = knn(x, k)
    u, cost = average_link_graph_segment(G, qmax=2)
    v = np.zeros(n)
    v[:int(0.7*n)]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def ward_test_basic(n=100,k=5):
    """ Basic check of ward's algorithm
    """
    np.random.seed(0)
    x = np.random.randn(n, 2)
    x[:int(0.7*n)] += 3
    G = knn(x, k)
    t = ward(G,x)
    u = t.split(2)
    v = np.zeros(n)
    v[:int(0.7*n)]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def wardq_test_basic(n=100,k=5):
    """ Basic check of ward's algorithm
    """
    np.random.seed(0)
    x = np.random.randn(n, 2)
    x[:int(0.7*n)] += 3
    G = knn(x, k)
    t = ward_quick(G, x)
    u = t.split(2)
    v = np.zeros(n)
    v[:int(0.7*n)]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def wardq_test_2():
    """
    Check that we hasnle the case of a graph with too many
    connected components
    """
    np.random.seed(0)
    n = 100
    k = 5
    x = np.random.randn(n, 2)
    x[:int(0.3*n)] += 10
    x[int(0.8*n):] -= 10
    G = knn(x, k)
    t = ward_quick(G, x)
    u = t.split(2)
    assert(u.max()==2)


def wardf_test(n=100,k=5):
    """
    """  
    np.random.seed(0)
    x = np.random.randn(n,2)
    x[:int(0.7*n)] += 3
    G = knn(x, 5)
    F = field_from_graph_and_data(G, x)
    u, cost = ward_field_segment(F, qmax=2)
    v = np.zeros(n)
    v[:int(0.7*n)]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def wards_test_basic(n=100,k=5):
    """ Basic check of ward's segmentation algorithm
    """
    np.random.seed(0)
    x = np.random.randn(n, 2)
    x[:int(0.7*n)] += 3
    G = knn(x, k)
    u,cost =  ward_segment(G, x, qmax=2)
    v = np.zeros(n)
    v[:int(0.7*n)]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def wards_test_3():
    """Check that ward_segment
    """
    np.random.seed(0)
    n = 100
    k = 5
    x = np.random.randn(n,2)
    x[:int(0.3*n)] += 10
    x[int(0.8*n):] -= 10
    G = knn(x,k)
    u,cost = ward_segment(G, x, qmax=2)
    assert(u.max()==2)

def cost_test(n=100, k=5):
    """
    check that cost.max() is equal to the data variance
    """
    np.random.seed(0)
    x = np.random.randn(n, 2)
    G = knn(x, k)
    u, cost =  ward_segment(G, x)
    print cost.max()/n, np.var(x, 0).sum()
    assert np.abs(cost.max()/(n*np.var(x,0).sum()) - 1)<1.e-6

def ward_test_more(n=100, k=5, verbose=0):
    """
    Check that two implementations give the same result
    """
    np.random.seed(0)
    X = randn(n,2)
    X[:np.ceil(n/3)] += 5
    G = knn(X, 5)
    u,c = ward_segment(G, X, stop=-1, qmax=1, verbose=verbose)
    u1,c = ward_segment(G, X, stop=-1, qmax=k, verbose=verbose)
    u,c = ward_quick_segment(G, X, stop=-1, qmax=1, verbose=verbose)
    u2,c = ward_quick_segment(G, X, stop=-1, qmax=k, verbose=verbose)
    assert(np.sum(u1==u2)==n)



if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
