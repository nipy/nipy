"""
Several basic tests
for hierarchical clustering procedures
should be cast soon in a nicer unitest framework

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
from numpy.random import rand, randn
from neuroimaging.neurospin.clustering.hierarchical_clustering import *
import neuroimaging.neurospin.graph as fg
from numpy.random import rand,randn

def ALC_test(n=10,verbose=0):
    """
    Simple test that performs an average link clustering of the data
    Not a real check
    INPUT:
    n=10: number of clustered items
    """
    X = np.arange(n)
    X = np.hstack((X,X+0.1*randn(np.size(X))));
    X = np.reshape(X,(np.size(X),1))
    t = Average_Link_Euclidian(X,1)
    u = t.partition(1.1)
    if verbose:
        print u

def alc_test(n=1000,k=2,verbose=0):
    """
    Simple test that performs an average link clustering of the data
    Not a real check
    INPUT
    n=1000: number of clustered items
    k=2: number of desired clusters
    """
    X = randn(n,2)
    G = fg.WeightedGraph(n)
    G.knn(X,k)
    u,cost = Average_Link_Graph_segment(G,stop=0)
    v = G.cc()
    if verbose:
        print u.max(),v.max()
        import matplotlib.pylab as mp
        for i in range(u.max()+1):
            mp.plot(X[u==i,0],X[u==i,1],'o',color=(rand(),rand(),rand()))
        mp.show()

def ward_new_test(n=100,k=5, verbose=0):
    """
    basic testing for ward's algorithm.
    INPUT:
    n=100: number of items to cluster
    k=5: number of desired clusters
    """
    X = randn(n,2)
    X[:np.ceil(n/3)] += 5
    G = fg.WeightedGraph(n)
    #G.mst(X)
    G.knn(X,5)
    t = Ward(G,X,verbose)
    if verbose:
        print t.check_compatible_height()
        t.plot()
        t.plot_height()

    u = t.partition(1.0)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        for i in range(u.max()+1):
            mp.plot(X[u==i,0],X[u==i,1],'o',color=(rand(),rand(),rand()))
        mp.show()

    u = t.split(k)
    if verbose:
        mp.figure()
        for i in range(u.max()+1):
            mp.plot(X[u==i,0],X[u==i,1],'o',color=(rand(),rand(),rand()))
        mp.show()

def ward_test(n=100,k=5,verbose=0):
    """
    basic testing for old ward's API
    INPUT:
    n=100: number of items to cluster
    k=5: number of desired clusters  
    """
    from numpy.random import rand,randn
    X = randn(n,2)
    X[:np.ceil(n/3)] += 5
    G = fg.WeightedGraph(n)
    #G.mst(X)
    G.knn(X,5)
    import time
    t1 = time.time()
    u,c = Ward_segment(G,X,stop=-1,qmax=1,verbose=verbose)
    t2 = time.time()
    u1,c = Ward_segment(G,X,stop=-1,qmax=k,verbose=verbose)
    t3 = time.time()
    u,c = Ward_quick_segment(G,X,stop=-1,qmax=1,verbose=verbose)
    t4 = time.time()
    u2,c = Ward_quick_segment(G,X,stop=-1,qmax=k,verbose=verbose)
    if verbose:
        print t4-t3,t2-t1,np.sum(u1==u2)
        import matplotlib.pylab as mp
        mp.figure()
        for i in range(u2.max()+1):
            mp.plot(X[u2==i,0],X[u2==i,1],'o',color=(rand(),rand(),rand()))
        mp.show()


def max_distance_test(n=100,k=5,verbose=0):
    """
    basic testing of distance-based maximum link clustering
    INPUT:
    n=100: number of items to cluster
    k=5: number of desired clusters
    """
    X = randn(n,2)
    X[:np.ceil(n/3)] += 2
    D = Euclidian_distance(X)
    t = Maximum_Link_Distance(D)
    u,c = Maximum_Link_Distance_segment (D,stop=-1,qmax=k)

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        for i in range(k):
            mp.plot(X[u==i,0],X[u==i,1],'o',color=(rand(),rand(),rand()))
        mp.show()


def av_distance_test(n=100,k=5,verbose=0):
    """
    basic testing of distance_based average link clustering
    INPUT:
    n=100: number of items to cluster
    k=5: number of desired clusters
    """    
    X = randn(n,2)
    X[:np.ceil(n/3)] += 2
    D = Euclidian_distance(X)
    u,c = Average_Link_Distance_segment(D,stop=-1,verbose=verbose)
    u,c = Average_Link_Distance_segment(D,stop=-1,qmax=k,verbose=verbose)

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        for i in range(k):
            mp.plot(X[u==i,0],X[u==i,1],'o',color=(rand(),rand(),rand()))
        mp.show()

def avlink_graph_test(n=100,k=5,verbose=0):
    """
    basic testing of graph_based avergae link clustering
    INPUT:
    n=100: number of items to cluster
    k=5: number of desired clusters
    """
    X = randn(n,2)
    X[:np.ceil(n/3)] += 2
    G = fg.WeightedGraph(n)
    G.knn(X,5)
    G.set_gaussian(X)
    u,c = Average_Link_Graph_segment(G,stop=-1,qmax=k)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        for i in range(k):
            mp.plot(X[u==i,0],X[u==i,1],'o',color=(rand(),rand(),rand()))
        mp.show()

def ward_simple_test(n=100,k=2,verbose=0):
    """
    basic testing of ward clustering
    INPUT:
    n=100: number of items to cluster
    k=5: number of desired clusters
    """
    X = randn(n,2)
    X[:np.ceil(n/3)] += 2
    t = Ward_simple(X)
    u = t.split(k)
    if verbose:
        t.plot()
        t.plot_height()
       
        import matplotlib.pylab as mp
        mp.figure()
        for i in range(u.max()+1):
            mp.plot(X[u==i,0],X[u==i,1],'o',color=(rand(),rand(),rand()))
        mp.show()
        print t.list_of_subtrees()


def ward_test_idiot(n=100,k=2,verbose=0):
    """
    basic testing of ward clustering
    """
    X = randn(n,2)
    X[:np.ceil(n/3)] += 2
    t = Ward_simple(X)
    validleaves = np.zeros(n)
    validleaves[:np.ceil(n/3)]=1

    n = np.sum(t.isleaf())
    valid = np.zeros(t.V,'bool')
    valid[t.isleaf()]=validleaves.astype('bool')
    nv =  np.sum(validleaves)
    nv0 = 0
    while nv>nv0:
        nv0= nv
        for v in range(t.V):
            if valid[v]:
                valid[t.parents[v]]=1
        nv = np.sum(valid)
    
    if verbose:
        import matplotlib.pylab as mp
        t.fancy_plot_(valid)

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
