#!/usr/bin/env python

import numpy as np
import numpy.random as nr
from unittest import TestCase

from ..graph import (WeightedGraph, complete_graph, mst, knn, eps_nn, 
                     wgraph_from_adjacency, wgraph_from_coo_matrix, 
                     concatenate_graphs, wgraph_from_3d_grid)


def basicdata():
    x = np.array( [[-1.998,-2.024], [-0.117,-1.010], [1.099,-0.057],
                   [ 1.729,-0.252], [1.003,-0.021], [1.703,-0.739],
                   [-0.557,1.382],[-1.200,-0.446],[-0.331,-0.256],
                   [-0.800,-1.584]])
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

class test_Graph(TestCase):
    
    def test_complete(self):
        v = 10
        G = complete_graph(v)
        a = G.get_edges()[:, 0]
        b = G.get_edges()[:, 1]
        inds = np.indices((v, v)).reshape( (2, v * v) )
        self.assert_( ( inds == (a, b) ).all() )
  
    def test_knn_1(self):
        x = basicdata()
        G = knn(x, 1)
        A = G.get_edges()[:,0]
        OK = (np.shape(A)[0] == (14))
        self.assert_(OK)
    
    def test_set_euclidian(self):
        G,x = basic_graph_2()
        d = G.weights
        G.set_euclidian(x / 10)
        D = G.weights
        OK = np.allclose(D, d / 10, 1e-7)
        self.assert_(OK)

    def test_set_gaussian(self):
        G,x = basic_graph_2()
        d = G.weights
        G.set_gaussian(x, 1.0)
        D = G.weights
        OK = np.allclose(D, np.exp(- d * d / 2), 1e-7)
        self.assert_(OK)

    def test_set_gaussian_2(self):
        G,x = basic_graph_2()
        d = G.weights
        G.set_gaussian(x)
        D = G.weights
        sigma = sum(d * d) / len(d)
        OK = np.allclose(D, np.exp(-d * d / (2 * sigma)), 1e-7)
        self.assert_(OK)

    def test_eps_1(self):
        x = basicdata()
        G = eps_nn(x, 1.)
        D = G.weights
        OK = (np.size(D) == 16)
        self.assert_(OK)
        OK = (D < 1).all()
        self.assert_(OK)

    def test_mst_1(self):
        x = basicdata()
        G = mst(x)
        D = G.weights
        OK = (np.size(D) == 18)
        self.assert_(OK)

    def test_3d_grid(self):
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
            assert wgraph_from_3d_grid(xyz, 6).E == 2
            assert wgraph_from_3d_grid(xyz, 18).E == 2
            assert wgraph_from_3d_grid(xyz, 26).E == 2
        for x in x2:
            xyz = np.vstack((x0, x))
            assert wgraph_from_3d_grid(xyz, 6).E == 0
            assert wgraph_from_3d_grid(xyz, 18).E == 2
            assert wgraph_from_3d_grid(xyz, 26).E == 2
        for x in x3:
            xyz = np.vstack((x0, x))
            assert wgraph_from_3d_grid(xyz, 6).E == 0
            assert wgraph_from_3d_grid(xyz, 18).E == 0
            assert wgraph_from_3d_grid(xyz, 26).E == 2
            

    def test_grid_3d_1(self):
        """ Test the 6 nn graphs on 3d grid
        """
        nx, ny, nz = 9, 6, 1
        xyz = np.mgrid[0:nx, 0:ny, 0:nz]
        xyz = np.reshape(xyz, (3, nx * ny * nz)).T
        G = wgraph_from_3d_grid(xyz, 6)
        self.assert_(G.E == 186)
    
    def test_grid_3d_2(self):
        """ Test the 18-nn graph on a 3d grid
        """
        nx, ny, nz = 9, 6, 1
        xyz = np.mgrid[0:nx, 0:ny, 0:nz]
        xyz = np.reshape(xyz,(3, nx * ny * nz)).T
        G = wgraph_from_3d_grid(xyz, 18)
        self.assert_(G.E == 346)
        
    def test_grid_3d_3(self):
        """ Test the 26-nn graph on a 3d grid
        """
        nx, ny, nz = 9, 6, 1
        xyz = np.mgrid[0:nx, 0:ny, 0:nz]
        xyz = np.reshape(xyz,(3, nx * ny * nz)).T
        G = wgraph_from_3d_grid(xyz, 26)
        self.assert_(G.E == 346)

    def test_grid_3d_4(self):
        nx, ny, nz = 10, 10, 10
        xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
        G = wgraph_from_3d_grid(xyz, 26)
        D = G.weights
        # 6 * 9 * 10 * 10
        self.assert_(sum(D == 1)==5400 )
        # 26 * 8 ** 3 + 6 * 8 ** 2 * 17 + 12 * 8 * 11 + 8 * 7 
        self.assert_(np.size(D) == 20952 )
        # 18 * 8 ** 3 + 6 * 8 ** 2 * 13 + 12 * 8 * 9 + 8 * 6
        self.assert_(sum(D < 1.5) == 15120)

    def test_grid_3d_5(self):
        nx, ny, nz = 5, 5, 5
        xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
        G = wgraph_from_3d_grid(xyz, 26)
        D = G.weights.copy()
        G.set_euclidian(xyz)
        assert (np.allclose(G.weights, D, 1.e-7))

    def test_grid_3d_6(self):
        nx, ny, nz = 5, 5, 5
        xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
        adj = wgraph_from_3d_grid(xyz, 26).to_coo_matrix().tolil()
        assert len(adj.rows[63]) == 26
        for i in [62, 64, 58, 68, 38, 88, 57, 67, 37, 87, 59, 69, 39, 89, 33, 
                  83, 43, 93, 32, 82, 42, 92, 34, 84, 44, 94]:
            assert i in adj.rows[63]

    def test_grid_3d_7(self):
        """ Check that the grid graph is symmetric
        """
        xyz = np.array(np.where(np.random.rand(5, 5, 5) > 0.5)).T
        adj = wgraph_from_3d_grid(xyz, 6).to_coo_matrix()
        assert (adj - adj.T).nnz == 0
        adj = wgraph_from_3d_grid(xyz, 18).to_coo_matrix()
        assert (adj - adj.T).nnz == 0
        adj = wgraph_from_3d_grid(xyz, 26).to_coo_matrix()
        assert (adj - adj.T).nnz == 0
        

    def test_cut_redundancies(self):
        G = basic_graph()
        e = G.E
        edges = G.get_edges()
        weights = G.weights
        G.E = 2 * G.E
        G.edges = np.concatenate((edges, edges))
        G.weights = np.concatenate((weights, weights))
        K = G.cut_redundancies()
        OK = (K.E == e)
        self.assert_(OK)

    def test_degrees(self):
        G = basic_graph()
        (r,l) = G.degrees()
        self.assert_(( r == 2 ).all())
        self.assert_(( l == 2 ).all())

    def test_normalize(self):
        G = basic_graph()
        G.normalize()
        M = G.to_coo_matrix()
        sM = np.array(M.sum(1)).ravel()
        test = np.absolute(sM - 1) < 1.e-7
        OK = np.size(np.nonzero(test) == 0)
        self.assert_(OK)

    def test_normalize_2(self):
        G = basic_graph()
        G.normalize(0)
        M = G.to_coo_matrix()
        sM = np.array(M.sum(1)).ravel()
        test = np.absolute(sM - 1) < 1.e-7
        OK = np.size(np.nonzero(test)==0)
        self.assert_(OK)

    def test_normalize_3(self):
        G = basic_graph()
        G.normalize(1)
        M = G.to_coo_matrix()
        sM = np.array(M.sum(0)).ravel()
        test = np.absolute(sM - 1) < 1.e-7
        OK = np.size(np.nonzero(test)==0)
        self.assert_(OK)

    def test_adjacency(self):
        G = basic_graph()
        M = G.to_coo_matrix()
        self.assert_(( M.diagonal() == 0 ).all())
        A = M.toarray()
        self.assert_(( np.diag(A, 1) != 0 ).all())
        self.assert_(( np.diag(A, -1) != 0 ).all())       

    def test_cc(self):
        G = basic_graph()
        l = G.cc()
        L = np.array(l==0)
        OK = L.all()
        self.assert_(OK)

    def test_isconnected(self):
        G = basic_graph()
        self.assert_(G.is_connected())

    def test_main_cc(self):
        x = basicdata()
        G = knn(x, 1)
        l = G.cc()
        l = G.main_cc()
        assert np.size(l)==6

    def test_dijkstra(self):
        """ Test dijkstra's algorithm
        """
        G = basic_graph()
        l = G.dijkstra(0)
        assert (np.absolute(l[10] - 20 * np.sin(np.pi / 20)) < 1.e-7)

    def test_dijkstra_multiseed(self):
        """ Test dijkstra's algorithm, multi_seed version
        """
        G = basic_graph()
        l = G.dijkstra([0, 1])
        assert (np.absolute(l[10] - 18 * np.sin(np.pi / 20)) < 1.e-7)


    def test_dijkstra2(self):
        """ Test dijkstra's algorithm, API detail
        """
        G = basic_graph()
        l = G.dijkstra()
        assert (np.absolute(l[10] - 20 * np.sin(np.pi / 20)) < 1.e-7)
        
    def test_compact_representation(self):
        """ Test that the compact representation of the graph is indeed correct
        """
        G = basic_graph()
        idx, ne, we = G.compact_neighb()
        assert len(idx) == 21
        assert idx[0] == 0
        assert idx[20] == G.E
        assert len(ne) == G.E
        assert len(we) == G.E

    def test_floyd_1(self):
        """ Test Floyd's algo without seed
        """
        G = basic_graph()
        l = G.floyd()
        for i in range(10):
            plop = np.absolute(np.diag(l, i) - 2 * i * np.sin(2 * np.pi / 40))
            assert(plop.max()<1.e-4)

    def test_floyd_2(self):
        """ Test Floyd's algo, with seed
        """
        G = basic_graph()
        seeds = np.array([0,10])
        l = G.floyd(seeds)
        
        for i in range(10):
            plop = np.absolute(l[0,i]-2*i*np.sin(2*np.pi/40))
            assert (plop.max()<1.e-4)
            plop = np.absolute(l[0,19-i]-2*(i+1)*np.sin(2*np.pi/40))
            assert (plop.max()<1.e-4)

        for i in range(10):
            plop = np.absolute(l[1,i]-2*(10-i)*np.sin(2*np.pi/40))
            assert (plop.max()<1.e-4)
            plop = np.absolute(l[1,19-i]-2*(9-i)*np.sin(2*np.pi/40))
            assert (plop.max()<1.e-4)
  
    def test_symmeterize(self):
        a = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        b = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 0, 0, 1])
        edges = np.vstack((a, b)).T
        d = np.ones(14)
        G = WeightedGraph(7, edges, d)
        G.symmeterize()
        d = G.weights
        ok = (d == 0.5)
        self.assert_(ok.all())

    def test_voronoi(self):
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
        assert(label[1] == 0)
        
    def test_voronoi2(self):
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
        assert(label[4] == 0)
 
    def test_voronoi3(self):
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
        assert(label[4] == - 1)

    def test_concatenate1(self,n=10,verbose=0):
        x1 = nr.randn(n,2) 
        x2 = nr.randn(n,2) 
        G1 = knn(x1, 5)
        G2 = knn(x2, 5) 
        G = concatenate_graphs(G1, G2)
        if verbose:
            G.plot(np.hstack((x1, x2)))
        self.assert_(G.cc().max()>0)

    def test_concatenate2(self,n=10,verbose=0):
        G1 = complete_graph(n)
        G2 = complete_graph(n)
        G = concatenate_graphs(G1, G2)
        self.assert_(G.cc().max() == 1)

    def test_anti_symmeterize(self,verbose=0):
        n = 10
        eps = 1.e-7
        M = (nr.rand(n, n) > 0.7).astype(np.float) 
        C = M - M.T
        G = wgraph_from_adjacency(M)
        G.anti_symmeterize()
        A = G.to_coo_matrix()
        self.assert_(np.sum(C - A) ** 2 < eps)

    def test_subgraph_1(self,n=10,verbose=0):
        x = nr.randn(n, 2) 
        G = WeightedGraph(x.shape[0])
        valid = np.zeros(n)
        g = G.subgraph(valid)
        self.assert_(g is None)

    def test_subgraph_2(self,n=10,verbose=0):
        x = nr.randn(n,2) 
        G = knn(x, 5)
        valid = np.zeros(n)
        valid[:n/2] = 1
        g = G.subgraph(valid)
        self.assert_(g.edges.max() < n / 2)

    def test_graph_create_from_array(self):
        """
        Test the creation of a graph from a sparse coo_matrix 
        """
        a = np.random.randn(5, 5)
        wg = wgraph_from_adjacency(a)
        b = wg.to_coo_matrix()
        self.assert_((a == b.todense()).all())
        
    def test_graph_create_from_coo_matrix(self):
        """
        Test the creation of a graph from a sparse coo_matrix 
        """
        import scipy.sparse as spp
        a = (np.random.randn(5, 5) > .8).astype(np.float)
        s = spp.coo_matrix(a)
        wg = wgraph_from_coo_matrix(s)
        b = wg.to_coo_matrix()
        self.assert_((b.todense() == a).all())

    def test_to_coo_matrix(self):
        """ Test the generation of a sparse matrix as output 
        """
        a = (np.random.randn(5, 5)>.8).astype(np.float)
        wg = wgraph_from_adjacency(a)
        b = wg.to_coo_matrix().todense()
        self.assert_((a==b).all())
    
    def test_list_neighbours(self):
        """ test the generation of neighbours list
        """
        bg = basic_graph()
        nl = bg.list_of_neighbors()
        assert(len(nl) == bg.V)
        for ni in nl:
            assert len(ni)== 2
    
    def test_kruskal(self):
        """ test Kruskal's algor to thin the graph
        """
        x = basicdata()
        dmax = np.sqrt((x ** 2).sum())
        m = mst(x)
        g = eps_nn(x, dmax)
        k = g.kruskal()
        assert np.abs(k.weights.sum() - m.weights.sum() < 1.e-7)

    def test_concatenate3(self):
        """ test the graph concatenation utlitity
        """
        bg = basic_graph()
        cg = concatenate_graphs(bg, bg)
        valid = np.zeros(cg.V)
        valid[:bg.V] = 1
        sg = cg.subgraph(valid)
        assert (sg.edges == bg.edges).all()
        assert (sg.weights == bg.weights).all()

    def test_cliques(self):
        """ test the computation of cliques
        """
        x = np.random.rand(20, 2)
        x[15:] += 2.
        g = knn(x, 5)
        g.set_gaussian(x, 1.)
        cliques = g.cliques()
        assert len(np.unique(cliques)) > 1

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


