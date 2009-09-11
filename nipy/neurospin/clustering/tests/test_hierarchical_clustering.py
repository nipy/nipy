"""
Several basic tests
for hierarchical clustering procedures
should be cast soon in a nicer unitest framework

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
from numpy.random import rand, randn
from nipy.neurospin.clustering.hierarchical_clustering import *
import nipy.neurospin.graph as fg
import nipy.neurospin.graph.field as ff
from numpy.random import rand,randn

def alc_test_basic():
    np.random.seed(0)
    x = np.random.randn(10,2)
    x[:7,0] += 3
    t = average_link_euclidian(x,1)
    u = t.split(2)
    v = np.zeros(10)
    v[:7]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def mlc_test_basic():
    np.random.seed(0)
    x = np.random.randn(10,2)
    x[:7,0] += 3
    t = maximum_link_euclidian(x,1)
    u = t.split(2)
    v = np.zeros(10)
    v[:7]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)
    
def ward_nograph_test_basic1(n=100,k=5):
    """
    Check that we obtain the correct solution in a simplistic case 
    """
    np.random.seed(0)
    x = np.random.randn(n,2)
    x[:int(0.7*n)] += 3
    t = ward_simple(x)
    u = t.split(2)
    v = np.zeros(n)
    v[:int(0.7*n)]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def alcd_test_basic():
    np.random.seed(0)
    x = np.random.randn(10,2)
    x[:7,0] += 3
    dist = np.array([[np.sqrt(np.sum((x[i]-x[j])**2))
                      for i in range(10)] for j in range(10)])
    u,cost = average_link_distance_segment(dist,qmax=2)
    v = np.zeros(10)
    v[:7]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def mlcd_test_basic():
    np.random.seed(0)
    x = np.random.randn(10,2)
    x[:7,0] += 3
    dist = np.array([[np.sqrt(np.sum((x[i]-x[j])**2))
                      for i in range(10)] for j in range(10)])
    u,cost = maximum_link_distance_segment(dist,qmax=2)
    v = np.zeros(10)
    v[:7]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def alg_test_basic(n=100,k=5):
    """
    Check that we obtain the correct solution in a simplistic case 
    """
    np.random.seed(0)
    x = np.random.randn(n,2)
    x[:int(0.7*n)] += 3
    G = fg.WeightedGraph(n)
    G.knn(x,k)
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
    x = np.random.randn(n,2)
    x[:int(0.3*n)] += 10
    x[int(0.8*n):] -= 10
    G = fg.WeightedGraph(n)
    G.knn(x,k)
    t = average_link_graph(G)
    u = t.split(2)
    assert(u.max()==2)

def alg_test_3(n=100,k=5):
    """
    Check that we obtain the correct solution in a simplistic case 
    """
    np.random.seed(0)
    x = np.random.randn(n,2)
    x[:int(0.7*n)] += 3
    G = fg.WeightedGraph(n)
    G.knn(x,k)
    u, cost = average_link_graph_segment(G,qmax=2)
    v = np.zeros(n)
    v[:int(0.7*n)]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def ward_test_basic(n=100,k=5):
    """ Basic check of ward's algorithm
    """
    np.random.seed(0)
    x = np.random.randn(n,2)
    x[:int(0.7*n)] += 3
    G = fg.WeightedGraph(n)
    G.knn(x,k)
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
    x = np.random.randn(n,2)
    x[:int(0.7*n)] += 3
    G = fg.WeightedGraph(n)
    G.knn(x,k)
    t = ward_quick(G,x)
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
    x = np.random.randn(n,2)
    x[:int(0.3*n)] += 10
    x[int(0.8*n):] -= 10
    G = fg.WeightedGraph(n)
    G.knn(x,k)
    t = ward_quick(G,x)
    u = t.split(2)
    assert(u.max()==2)


def wardf_test(n=100,k=5):
    """
    """  
    np.random.seed(0)
    x = np.random.randn(n,2)
    x[:int(0.7*n)] += 3
    F = ff.Field(n)
    F.knn(x,5)
    F.set_field(x)
    u,cost = ward_field_segment(F,qmax=2)
    v = np.zeros(n)
    v[:int(0.7*n)]=1
    w = np.absolute(u-v)
    assert(np.sum(w*(1-w))==0)

def wards_test_basic(n=100,k=5):
    """ Basic check of ward's segmentation algorithm
    """
    np.random.seed(0)
    x = np.random.randn(n,2)
    x[:int(0.7*n)] += 3
    G = fg.WeightedGraph(n)
    G.knn(x,k)
    u,cost =  ward_segment(G,x, qmax=2)
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
    G = fg.WeightedGraph(n)
    G.knn(x,k)
    u,cost = ward_segment(G,x,qmax=2)
    assert(u.max()==2)

def cost_test(n=100,k=5):
    """
    check that cost.max() is equal to the data variance
    """
    np.random.seed(0)
    x = np.random.randn(n,2)
    G = fg.WeightedGraph(n)
    G.knn(x,k)
    u,cost =  ward_segment(G,x)
    assert(cost.max()==np.var(x,0).sum())

def ward_test_more(n=100,k=5,verbose=0):
    """
    Check that two implementations give the same result
    """
    np.random.seed(0)
    X = randn(n,2)
    X[:np.ceil(n/3)] += 5
    G = fg.WeightedGraph(n)
    G.knn(X,5)
    import time
    t1 = time.time()
    u,c = ward_segment(G,X,stop=-1,qmax=1,verbose=verbose)
    t2 = time.time()
    u1,c = ward_segment(G,X,stop=-1,qmax=k,verbose=verbose)
    t3 = time.time()
    u,c = ward_quick_segment(G,X,stop=-1,qmax=1,verbose=verbose)
    t4 = time.time()
    u2,c = ward_quick_segment(G,X,stop=-1,qmax=k,verbose=verbose)
    if verbose:
        print t4-t3,t2-t1
    assert(np.sum(u1==u2)==n)



if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
