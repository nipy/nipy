#!/usr/bin/env python

import numpy as np
import numpy.random as nr
from numpy.testing import(assert_array_equal, assert_array_almost_equal,
                          assert_almost_equal)
from nose.tools import assert_true, assert_equal

from ..graph import (WeightedGraph, complete_graph, mst, knn, eps_nn, 
                     wgraph_from_adjacency, wgraph_from_coo_matrix, 
                     concatenate_graphs, wgraph_from_3d_grid)



def basicdata():
    x = np.array( [[- 1.998, - 2.024], [- 0.117, - 1.010], [1.099, - 0.057],
                   [ 1.729, - 0.252], [1.003, - 0.021], [1.703, - 0.739],
                   [- 0.557, 1.382],[- 1.200, - 0.446],[- 0.331, - 0.256],
                   [- 0.800, - 1.584]])
    return x


def basic_graph():
    l = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    x = np.column_stack((np.cos(l), np.sin(l)))
    G = knn(x, 2)
    return G


def basic_graph_2():
    l = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    x = np.column_stack((np.cos(l), np.sin(l)))
    G = knn(x, 2)
    return G, x


def test_complete():
    v = 10
    G = complete_graph(v)
    a = G.get_edges()[:, 0]
    b = G.get_edges()[:, 1]
    inds = np.indices((v, v)).reshape( (2, v * v) )
    assert_array_equal(inds, (a, b))


def test_knn_1():
    x = basicdata()
    G = knn(x, 1)
    A = G.get_edges()[:, 0]
    assert_equal(np.shape(A)[0], 14)

    
def test_set_euclidian():
    G, x = basic_graph_2()
    d = G.weights
    G.set_euclidian(x / 10)
    D = G.weights
    assert_true(np.allclose(D, d / 10, 1e-7))


def test_set_gaussian():
    G, x = basic_graph_2()
    d = G.weights
    G.set_gaussian(x, 1.0)
    D = G.weights
    assert_true(np.allclose(D, np.exp(- d * d / 2), 1e-7))


def test_set_gaussian_2():
    G, x = basic_graph_2()
    d = G.weights
    G.set_gaussian(x)
    D = G.weights
    sigma = np.sum(d * d) / len(d)
    assert_true(np.allclose(D, np.exp(-d * d / (2 * sigma)), 1e-7))
    

def test_eps_1():
    x = basicdata()
    G = eps_nn(x, 1.)
    D = G.weights
    assert_equal(np.size(D), 16)
    assert_true((D < 1).all())

    
def test_mst_1():
    x = basicdata()
    G = mst(x)
    D = G.weights
    assert_equal(np.size(D), 18)


def test_3d_grid():
    """test the 6nn graph
    """
    x0 = np.array([0, 0, 0])
    x1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0],
                   [0, 0, -1]])
    x2 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, -1, 0], [0, 1, -1],
                   [1, 0, -1], [-1, 1, 0], [0, -1, 1], [-1, 0, 1], 
                   [-1, -1, 0], [-1, 0, -1], [0, -1, -1]])
    x3 = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                   [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
    for x in x1:
        xyz = np.vstack((x0, x))
        assert_equal(wgraph_from_3d_grid(xyz, 6).E, 2)
        assert_equal(wgraph_from_3d_grid(xyz, 18).E, 2)
        assert_equal(wgraph_from_3d_grid(xyz, 26).E, 2)
        for x in x2:
            xyz = np.vstack((x0, x))
            assert_equal(wgraph_from_3d_grid(xyz, 6).E, 0)
            assert_equal(wgraph_from_3d_grid(xyz, 18).E, 2)
            assert_equal(wgraph_from_3d_grid(xyz, 26).E, 2)
        for x in x3:
            xyz = np.vstack((x0, x))
            assert_equal(wgraph_from_3d_grid(xyz, 6).E, 0)
            assert_equal(wgraph_from_3d_grid(xyz, 18).E, 0)
            assert_equal(wgraph_from_3d_grid(xyz, 26).E, 2)
            

def test_grid_3d_1():
    """ Test the 6 nn graphs on 3d grid
    """
    nx, ny, nz = 9, 6, 1
    xyz = np.mgrid[0:nx, 0:ny, 0:nz]
    xyz = np.reshape(xyz, (3, nx * ny * nz)).T
    G = wgraph_from_3d_grid(xyz, 6)
    assert_equal(G.E, 186)

    
def test_grid_3d_2():
    """ Test the 18-nn graph on a 3d grid
    """
    nx, ny, nz = 9, 6, 1
    xyz = np.mgrid[0:nx, 0:ny, 0:nz]
    xyz = np.reshape(xyz,(3, nx * ny * nz)).T
    G = wgraph_from_3d_grid(xyz, 18)
    assert_equal(G.E, 346)
        

def test_grid_3d_3():
    """ Test the 26-nn graph on a 3d grid
    """
    nx, ny, nz = 9, 6, 1
    xyz = np.mgrid[0:nx, 0:ny, 0:nz]
    xyz = np.reshape(xyz,(3, nx * ny * nz)).T
    G = wgraph_from_3d_grid(xyz, 26)
    assert_equal(G.E, 346)


def test_grid_3d_4():
    nx, ny, nz = 10, 10, 10
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    G = wgraph_from_3d_grid(xyz, 26)
    D = G.weights
    # 6 * 9 * 10 * 10
    assert_equal(sum(D == 1), 5400 )
    # 26 * 8 ** 3 + 6 * 8 ** 2 * 17 + 12 * 8 * 11 + 8 * 7 
    assert_equal(np.size(D), 20952 )
    # 18 * 8 ** 3 + 6 * 8 ** 2 * 13 + 12 * 8 * 9 + 8 * 6
    assert_equal(sum(D < 1.5), 15120)


def test_grid_3d_5():
    nx, ny, nz = 5, 5, 5
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    G = wgraph_from_3d_grid(xyz, 26)
    D = G.weights.copy()
    G.set_euclidian(xyz)
    assert_array_almost_equal(G.weights, D)


def test_grid_3d_6():
    nx, ny, nz = 5, 5, 5
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    adj = wgraph_from_3d_grid(xyz, 26).to_coo_matrix().tolil()
    assert_equal(len(adj.rows[63]), 26)
    for i in [62, 64, 58, 68, 38, 88, 57, 67, 37, 87, 59, 69, 39, 89, 33, 
              83, 43, 93, 32, 82, 42, 92, 34, 84, 44, 94]:
        assert_true(i in adj.rows[63])


def test_grid_3d_7():
    """ Check that the grid graph is symmetric
    """
    xyz = np.array(np.where(np.random.rand(5, 5, 5) > 0.5)).T
    adj = wgraph_from_3d_grid(xyz, 6).to_coo_matrix()
    assert_equal((adj - adj.T).nnz, 0)
    adj = wgraph_from_3d_grid(xyz, 18).to_coo_matrix()
    assert_equal((adj - adj.T).nnz, 0)
    adj = wgraph_from_3d_grid(xyz, 26).to_coo_matrix()
    assert_equal((adj - adj.T).nnz, 0)
        
    
def test_cut_redundancies():
    G = basic_graph()
    e = G.E
    edges = G.get_edges()
    weights = G.weights
    G.E = 2 * G.E
    G.edges = np.concatenate((edges, edges))
    G.weights = np.concatenate((weights, weights))
    K = G.cut_redundancies()
    assert_equal(K.E, e)


def test_degrees():
    G = basic_graph()
    (r, l) = G.degrees()
    assert_true((r == 2).all())
    assert_true((l == 2).all())


def test_normalize():
    G = basic_graph()
    G.normalize()
    M = G.to_coo_matrix()
    sM = np.array(M.sum(1)).ravel()
    assert_true((np.abs(sM - 1) < 1.e-7).all())
    

def test_normalize_2():
    G = basic_graph()
    G.normalize(0)
    M = G.to_coo_matrix()
    sM = np.array(M.sum(1)).ravel()
    assert_true((np.abs(sM - 1) < 1.e-7).all())
    

def test_normalize_3():
    G = basic_graph()
    G.normalize(1)
    M = G.to_coo_matrix()
    sM = np.array(M.sum(0)).ravel()
    assert_true((np.abs(sM - 1) < 1.e-7).all())
    

def test_adjacency():
    G = basic_graph()
    M = G.to_coo_matrix()
    assert_true(( M.diagonal() == 0 ).all())
    A = M.toarray()
    assert_true(( np.diag(A, 1) != 0 ).all())
    assert_true(( np.diag(A, -1) != 0 ).all())       


def test_cc():
    G = basic_graph()
    l = G.cc()
    L = np.array(l==0)
    assert_true(L.all())

    
def test_isconnected():
    G = basic_graph()
    assert_true(G.is_connected())
    

def test_main_cc():
    x = basicdata()
    G = knn(x, 1)
    l = G.cc()
    l = G.main_cc()
    assert_equal(np.size(l), 6)

def test_dijkstra():
    """ Test dijkstra's algorithm
    """
    G = basic_graph()
    l = G.dijkstra(0)
    assert_true(np.abs(l[10] - 20 * np.sin(np.pi / 20)) < 1.e-7)

def test_dijkstra_multiseed():
    """ Test dijkstra's algorithm, multi_seed version
    """
    G = basic_graph()
    l = G.dijkstra([0, 1])
    assert_true(np.abs(l[10] - 18 * np.sin(np.pi / 20)) < 1.e-7)


def test_dijkstra2():
    """ Test dijkstra's algorithm, API detail
    """
    G = basic_graph()
    l = G.dijkstra()
    assert_true(np.abs(l[10] - 20 * np.sin(np.pi / 20)) < 1.e-7)


def test_compact_representation():
    """ Test that the compact representation of the graph is indeed correct
    """
    G = basic_graph()
    idx, ne, we = G.compact_neighb()
    assert_equal(len(idx), 21)
    assert_equal(idx[0], 0)
    assert_equal(idx[20], G.E)
    assert_equal(len(ne), G.E)
    assert_equal(len(we), G.E)


def test_floyd_1():
    """ Test Floyd's algo without seed
    """
    G = basic_graph()
    l = G.floyd()
    for i in range(10):
        plop = np.abs(np.diag(l, i) - 2 * i * np.sin(2 * np.pi / 40))
        assert_true(plop.max() < 1.e-4)
        
def test_floyd_2():
    """ Test Floyd's algo, with seed
    """
    G = basic_graph()
    seeds = np.array([0,10])
    l = G.floyd(seeds)
    
    for i in range(10):
        plop = np.abs(l[0, i] - 2 * i * np.sin(2 * np.pi / 40))
        assert_true(plop.max() < 1.e-4)
        plop = np.abs(l[0,19 - i] - 2 * (i + 1) * np.sin(2 * np.pi / 40))
        assert_true(plop.max() < 1.e-4)

    for i in range(10):
        plop = np.abs(l[1, i] - 2 * (10 - i) * np.sin(2 * np.pi / 40))
        assert_true(plop.max() < 1.e-4)
        plop = np.abs(l[1, 19 - i] - 2 * (9 - i) * np.sin(2 * np.pi / 40))
        assert_true(plop.max() < 1.e-4)
  
def test_symmeterize():
    a = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    b = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 0, 0, 1])
    edges = np.vstack((a, b)).T
    d = np.ones(14)
    G = WeightedGraph(7, edges, d)
    G.symmeterize()
    d = G.weights
    assert_true((d == 0.5).all())


def test_voronoi():
    """ test voronoi labelling with 2 seeds
    """
    a = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    b = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 0, 0, 1])
    d = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
    edges = np.transpose(np.vstack((a, b)))
    G = WeightedGraph(7, edges,d)
    G.symmeterize()
    seed = np.array([0, 6])
    label = G.voronoi_labelling(seed)
    assert_equal(label[1], 0)
    
    
def test_voronoi2():
    """ test voronoi labelling with one seed
    """
    a = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    b = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 0, 0, 1])
    d = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
    edges = np.vstack((a, b)).T
    G = WeightedGraph(7, edges,d)
    G.symmeterize()
    seed = np.array([0])
    label = G.voronoi_labelling(seed)
    assert_equal(label[4], 0)
 

def test_voronoi3():
    """ test voronoi labelling with non-connected components
    """
    a = np.array([0, 1, 2, 5, 6])
    b = np.array([1, 2, 3, 6, 0])
    d = np.array([1, 1, 1, 1, 1]);
    edges = np.vstack((a, b)).T
    G = WeightedGraph(7, edges,d)
    G.symmeterize()
    seed = np.array([0])
    label = G.voronoi_labelling(seed)
    assert_equal(label[4], - 1)

def test_concatenate1(n=10):
    x1 = nr.randn(n, 2) 
    x2 = nr.randn(n, 2) 
    G1 = knn(x1, 5)
    G2 = knn(x2, 5) 
    G = concatenate_graphs(G1, G2)
    assert_true(G.cc().max() > 0)


def test_concatenate2(n=10):
    G1 = complete_graph(n)
    G2 = complete_graph(n)
    G = concatenate_graphs(G1, G2)
    assert_true(G.cc().max() == 1)


def test_anti_symmeterize():
    n = 10
    eps = 1.e-7
    M = (nr.rand(n, n) > 0.7).astype(np.float) 
    C = M - M.T
    G = wgraph_from_adjacency(M)
    G.anti_symmeterize()
    A = G.to_coo_matrix()
    assert_true(np.sum(C - A) ** 2 < eps)


def test_subgraph_1(n=10):
    x = nr.randn(n, 2) 
    G = WeightedGraph(x.shape[0])
    valid = np.zeros(n)
    assert(G.subgraph(valid) is None)


def test_subgraph_2(n=10):
    x = nr.randn(n, 2) 
    G = knn(x, 5)
    valid = np.zeros(n)
    valid[:n / 2] = 1
    assert_true(G.subgraph(valid).edges.max() < n / 2)


def test_graph_create_from_array():
    """Test the creation of a graph from a sparse coo_matrix 
    """
    a = np.random.randn(5, 5)
    wg = wgraph_from_adjacency(a)
    b = wg.to_coo_matrix()
    assert_array_equal(a, b.todense())
        

def test_graph_create_from_coo_matrix():
    """Test the creation of a graph from a sparse coo_matrix 
    """
    import scipy.sparse as spp
    a = (np.random.randn(5, 5) > .8).astype(np.float)
    s = spp.coo_matrix(a)
    wg = wgraph_from_coo_matrix(s)
    b = wg.to_coo_matrix()
    assert_array_equal(b.todense(), a)


def test_to_coo_matrix():
    """ Test the generation of a sparse matrix as output 
    """
    a = (np.random.randn(5, 5)>.8).astype(np.float)
    wg = wgraph_from_adjacency(a)
    b = wg.to_coo_matrix().todense()
    assert_array_equal(a, b)


def test_list_neighbours():
    """ test the generation of neighbours list
    """
    bg = basic_graph()
    nl = bg.list_of_neighbors()
    assert_equal(len(nl), bg.V)
    for ni in nl:
        assert_equal(len(ni), 2)


def test_kruskal():
    """ test Kruskal's algor to thin the graph
    """
    x = basicdata()
    dmax = np.sqrt((x ** 2).sum())
    m = mst(x)
    g = eps_nn(x, dmax)
    k = g.kruskal()
    assert_almost_equal(k.weights.sum(), m.weights.sum())


def test_concatenate3():
    """ test the graph concatenation utlitity
    """
    bg = basic_graph()
    cg = concatenate_graphs(bg, bg)
    valid = np.zeros(cg.V)
    valid[:bg.V] = 1
    sg = cg.subgraph(valid)
    assert_array_equal(sg.edges,  bg.edges)
    assert_array_equal(sg.weights, bg.weights)


def test_cliques():
    """ test the computation of cliques
    """
    x = np.random.rand(20, 2)
    x[15:] += 2.
    g = knn(x, 5)
    g.set_gaussian(x, 1.)
    cliques = g.cliques()
    assert_true(len(np.unique(cliques)) > 1)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


