from __future__ import absolute_import
#!/usr/bin/env python

import numpy as np
import numpy.random as nr
from unittest import TestCase

from ..bipartite_graph import (BipartiteGraph, cross_knn, cross_eps, 
                               check_feature_matrices)


def basicdata():
    x = np.array( [[-1.998,-2.024], [-0.117,-1.010], [1.099,-0.057],
                   [ 1.729,-0.252], [1.003,-0.021], [1.703,-0.739],
                   [-0.557,1.382],[-1.200,-0.446],[-0.331,-0.256],
                   [-0.800,-1.584]])
    return x

def test_feature_matrices():
    """ test that feature matrices are correctly checked
    """
    x, y = nr.rand(10, 1), nr.rand(12)
    check_feature_matrices(x, y)
    check_feature_matrices(y, x)
    check_feature_matrices(x, x)
    check_feature_matrices(y, y)

def test_cross_knn_1():
    """ test the construction of k-nn bipartite graph
    """
    x = basicdata()
    G = cross_knn(x, x, 2)
    assert (G.E == 20)
        
def test_cross_knn_2():
    """ test the construction of k-nn bipartite graph
    """
    x = basicdata()
    G = cross_knn(x, x, 1)
    assert (G.E == 10)
    
def test_cross_eps_1():
    """ test the construction of eps-nn bipartite graph
    """
    x = basicdata()
    y = x + 0.1 * nr.randn(x.shape[0], x.shape[1])
    G = cross_eps(x, y, 1.)
    D = G.weights
    assert((D < 1).all())

def test_copy():
    """ test that the weighted graph copy is OK
    """
    x = basicdata()
    G = cross_knn(x, x, 2)
    K = G.copy()
    assert K.edges.shape == (20, 2)
    
def test_subraph_left():
    """ Extraction of the 'left subgraph'
    """
    x = basicdata()
    g = cross_knn(x, x, 2)
    valid = np.arange(10) < 7
    sl = g.subgraph_left(valid)
    assert sl.V == 7
    assert sl.W == 10
    assert sl.edges[:, 0].max() == 6
    
def test_subraph_left2():
    """ Extraction of the 'left subgraph', without renumb=False
    """
    x = basicdata()
    g = cross_knn(x, x, 2)
    valid = np.arange(10) < 7
    sl = g.subgraph_left(valid, renumb=False)
    assert sl.V == 10
    assert sl.W == 10
    assert sl.edges[:, 0].max() == 6

def test_subraph_right():
    """ Extraction of the 'right subgraph'
    """
    x = basicdata()
    g = cross_knn(x, x, 2)
    valid = np.arange(10) < 7
    sr = g.subgraph_right(valid)
    assert sr.W == 7
    assert sr.V == 10
    assert sr.edges[:, 1].max() == 6

def test_subraph_right2():
    """ Extraction of the 'right subgraph', with renumb = False
    """
    x = basicdata()
    g = cross_knn(x, x, 2)
    valid = np.arange(10) < 7
    sr = g.subgraph_right(valid, renumb = False)
    assert sr.W == 10
    assert sr.V == 10
    assert sr.edges[:, 1].max() == 6
    


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


