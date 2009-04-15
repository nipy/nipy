#!/usr/bin/env python

from numpy.testing import *
import numpy as np
import numpy.random as nr
import nipy.neurospin.graph as fg
from unittest import TestCase

def basicdata():
    x = np.array([[-1.998,-2.024],[-0.117,-1.010],[1.099,-0.057],[ 1.729,-0.252],[ 1.003,-0.021],[1.703,-0.739],[-0.557,1.382],[-1.200,-0.446],[-0.331,-0.256],[-0.800,-1.584]])
    return x

def basic_graph():
    x = np.zeros((20,2),float)
    l = np.linspace(0, 2*np.pi, 20, endpoint=False)
    x[:,0] = np.cos(l)
    x[:,1] = np.sin(l)

    G = fg.WeightedGraph(20)
    G.knn(x,2)
    return G

def basic_graph_2():
    l = np.linspace(0, 2*np.pi, 20, endpoint=False)
    x = np.column_stack((np.cos(l), np.sin(l)))

    G = fg.WeightedGraph(20)
    G.knn(x,2)
    return G,x


class test_Graph(TestCase):
    
    def test_complete(self):
        v = 20
        G = fg.WeightedGraph(v)
        G.complete()
        a = G.get_edges()[:,0]
        b = G.get_edges()[:,1]
        inds = np.indices((v, v)).reshape( (2, v*v) )
        self.assert_( ( inds == (a, b) ).all() )
        M = G.adjacency()
        Dm0 = (M + np.eye(v) == 1)
        OK = Dm0.all() 
        self.assert_(OK)
    
    
    def test_knn_1(self):
        x = basicdata()
        G = fg.WeightedGraph(x.shape[0])
        G.knn(x,1)
        A = G.get_edges()[:,0]
        OK = (np.shape(A)[0]==(14))
        self.assert_(OK)
    def test_set_euclidian(self):
        G,x = basic_graph_2()
        d = G.weights
        G.set_euclidian(x/10)
        D = G.weights
        OK = np.allclose(D, d/10, 1e-7)
        self.assert_(OK)

    def test_set_gaussian(self):
        G,x = basic_graph_2()
        d = G.weights
        G.set_gaussian(x,1.0)
        D = G.weights
        OK = np.allclose(D, np.exp(-d*d/2), 1e-7)
        self.assert_(OK)

    def test_set_gaussian_2(self):
        G,x = basic_graph_2()
        d = G.weights
        G.set_gaussian(x)
        D = G.weights
        sigma = sum(d*d)/len(d)
        OK = np.allclose(D, np.exp(-d*d/(2*sigma)), 1e-7)
        self.assert_(OK)

    def test_eps_1(self):
        x = basicdata()
        G = fg.WeightedGraph(x.shape[0])
        G.eps(x,1.)
        D = G.weights
        OK = (np.size(D)==16)
        self.assert_(OK)
        OK = (D < 1).all()
        self.assert_(OK)

    def test_mst_1(self):
        x = basicdata()
        G = fg.WeightedGraph(x.shape[0])
        G.mst(x)
        D = G.weights
        OK = (np.size(D)==18)
        self.assert_(OK)

    def test_cross_knn_1(self):
        x = basicdata()
        G = fg.BipartiteGraph(x.shape[0],x.shape[0])
        G.cross_knn(x,x,2)
        
        OK = (G.E==20)
        self.assert_(OK)
        
    def test_cross_knn_2(self):
        x = basicdata()
        G = fg.BipartiteGraph(x.shape[0],x.shape[0])
        G.cross_knn(x,x,1)
        OK = (G.E==10)
        self.assert_(OK)  

    def test_cross_eps_1(self):
        x = basicdata()
        G = fg.BipartiteGraph(x.shape[0],x.shape[0])
        y = x +0.1*nr.randn(x.shape[0],x.shape[1])
        G.cross_eps(x,y,1.)
        D = G.weights
        self.assert_((D < 1).all())

    def test_grid_3d_1(self):
        nx, ny, nz = 9, 6, 1
        xyz = np.mgrid[0:nx,0:ny,0:nz]
        XYZ = np.transpose(np.reshape(xyz,(3,nx*ny*nz)))
        G = fg.WeightedGraph(XYZ.shape[0])
        G.from_3d_grid(XYZ,6)
        self.assert_(G.E==240)

    def test_grid_3d_2(self):
        nx = 9;
        ny = 6
        nz = 1;
        xyz = np.mgrid[0:nx,0:ny,0:nz]
        XYZ = np.transpose(np.reshape(xyz,(3,nx*ny*nz)))
        G = fg.WeightedGraph(XYZ.shape[0])
        G.from_3d_grid(XYZ,18)
        self.assert_(G.E==400)
        
    def test_grid_3d_3(self):
        nx = 9;
        ny = 6
        nz = 1;
        xyz = np.mgrid[0:nx,0:ny,0:nz];
        XYZ = np.transpose(np.reshape(xyz,(3,nx*ny*nz)));
        G = fg.WeightedGraph(XYZ.shape[0])
        G.from_3d_grid(XYZ,26)
        self.assert_(G.E==400)

    def test_grid_3d_4(self):
        nx, ny, nz = 10, 10, 10
        XYZ = np.transpose(np.reshape(np.indices((nx,ny,nz)),(3,nx*ny*nz)))
        G = fg.WeightedGraph(XYZ.shape[0])
        G.from_3d_grid(XYZ,26)
        D = G.weights
        self.assert_( sum(D==1)==5400 )
        self.assert_( np.size(D)==21952 )
        self.assert_( sum(D<1.5)==16120 )

    def test_cut_redundancies(self):
        G = basic_graph()
        e = G.E;
        edges = G.get_edges()
        weights = G.weights
        G.E = 2*G.E
        G.edges = np.concatenate((edges,edges))
        G.weights =np.concatenate((weights,weights))
        G.cut_redundancies()
        OK = (G.E==e)
        self.assert_(OK)

    def test_degrees(self):
        G = basic_graph()
        (r,l) = G.degrees()
        self.assert_(( r == 2 ).all())
        self.assert_(( l == 2 ).all())

    def test_normalize(self):
        G = basic_graph()
        G.normalize()
        M = G.adjacency()
        sM = np.sum(M,1)
        test = np.absolute(sM-1)<1.e-7
        OK = np.size(np.nonzero(test)==0)
        self.assert_(OK)

    def test_normalize_2(self):
        G = basic_graph()
        G.normalize(0)
        M = G.adjacency()
        sM = np.sum(M,1)
        test = np.absolute(sM-1)<1.e-7
        OK = np.size(np.nonzero(test)==0)
        self.assert_(OK)

    def test_normalize_3(self):
        G = basic_graph()
        G.normalize(1)
        M = G.adjacency()
        sM = np.sum(M,0)
        test = np.absolute(sM-1)<1.e-7
        OK = np.size(np.nonzero(test)==0)
        self.assert_(OK)

    def test_adjacency(self):
        G = basic_graph()
        M = G.adjacency()
        self.assert_(( np.diag(M) == 0 ).all())
        self.assert_(( np.diag(M, 1) != 0 ).all())
        self.assert_(( np.diag(M, -1) != 0 ).all())       
        
    def test_reorder(self):
        x = basicdata() 
        G = fg.WeightedGraph(x.shape[0])
        G.knn(x,1)
        G.reorder(0)
        A = G.get_edges()[:,0]
        B = G.get_edges()[:,1]
        OK = (A[1]>=A[0])&(A[3]>=A[2])&(A[5]>=A[4])&(A[7]>=A[6])&(A[9]>=A[8])
        self.assert_(OK)
        G.reorder(1)
        A = G.get_edges()[:,0]
        B = G.get_edges()[:,1]
        OK = (B[1]>=B[0])&(B[3]>=B[2])&(B[5]>=B[4])&(B[7]>=B[6])&(B[9]>=B[8])
        self.assert_(OK)
        G.reorder(2)
        A = G.get_edges()[:,0]
        B = G.get_edges()[:,1]
        D = G.weights
        OK = (D[1]>=D[0])&(D[3]>=D[2])&(D[5]>=D[4])&(D[7]>=D[6])&(D[9]>=D[8])
        self.assert_(OK)

    def test_reorder_2(self):
        x = basicdata()
        G = fg.WeightedGraph(x.shape[0])
        G.knn(x,1)
        G.reorder()
        A = G.get_edges()[:,0]
        B = G.get_edges()[:,1]
        OK = (A[1]>=A[0])&(A[3]>=A[2])&(A[5]>=A[4])&(A[7]>=A[6])&(A[9]>=A[9])
        G.reorder(1)
        A = G.get_edges()[:,0]
        B = G.get_edges()[:,1]
        OK = OK&(B[1]>=B[0])&(B[3]>=B[2])&(B[5]>=B[4])&(B[7]>=B[6])&(B[9]>=B[9])
        G.reorder(2)
        A = G.get_edges()[:,0]
        B = G.get_edges()[:,1]
        D = G.weights
        OK = OK&(D[1]>=D[0])&(D[3]>=D[2])&(D[5]>=D[4])&(D[7]>=D[6])&(D[9]>=D[9])
        self.assert_(OK)
        

    def test_cc(self):
        G = basic_graph()
        l = G.cc()
        L = np.array(l==0)
        OK = L.all()
        self.assert_(OK)

    def test_isconnected(self):
        G = basic_graph()
        b = G.is_connected()
        self.assert_(True)

    def test_main_cc(self):
        x = basicdata()
        G = fg.WeightedGraph(x.shape[0])
        G.knn(x,1)
        l = G.cc()
        l = G.main_cc()
        OK = np.size(l)==6
        self.assert_(OK)

    def test_dijkstra(self):
        G = basic_graph()
        l = G.dijkstra(0)
        OK = (np.absolute(l[10]-20*np.sin(np.pi/20))<1.e-7)
        self.assert_(OK)

    def test_dijkstra2(self):
        G = basic_graph()
        l = G.dijkstra()
        OK = (np.absolute(l[10]-20*np.sin(np.pi/20))<1.e-7)
        self.assert_(OK)

    def test_floyd_1(self):
        G = basic_graph()
        l = G.floyd()
        OK = True
        for i in range(10):
            plop = np.absolute(np.diag(l,i)-2*i*np.sin(2*np.pi/40))
            OK = OK & (plop.max()<1.e-4)
        self.assert_(OK)

    def test_floyd_2(self):
        G = basic_graph()
        seeds = np.array([0,10])
        l = G.floyd(seeds)
        
        OK = True

        for i in range(10):
            plop = np.absolute(l[0,i]-2*i*np.sin(2*np.pi/40))
            OK = OK & (plop.max()<1.e-4)
            plop = np.absolute(l[0,19-i]-2*(i+1)*np.sin(2*np.pi/40))
            OK = OK & (plop.max()<1.e-4)

        for i in range(10):
            plop = np.absolute(l[1,i]-2*(10-i)*np.sin(2*np.pi/40))
            OK = OK & (plop.max()<1.e-4)
            plop = np.absolute(l[1,19-i]-2*(9-i)*np.sin(2*np.pi/40))
            OK = OK & (plop.max()<1.e-4)
        
        self.assert_(OK)
  
    def test_symmeterize(self):
        a = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6])
        b = np.array([1,2,2,3,3,4,4,5,5,6,6,0,0,1])
        edges = np.transpose(np.stack((a,b)))
        d = np.ones(14);
        G = fg.WeightedGraph(7, edges,d)
        G.symmeterize()
        d = G.weights
        ok = (d==0.5)
        OK = ok.all()
        self.assert_(True)

    def test_symmeterize(self):
        a = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6])
        b = np.array([1,2,2,3,3,4,4,5,5,6,6,0,0,1])
        edges = np.transpose(np.vstack((a,b)))
        d = np.ones(14);
        G = fg.WeightedGraph(7, edges,d)
        G.symmeterize()
        ok = (d==0.5)
        OK = ok.all()
        self.assert_(True)

    def test_voronoi(self):
        a = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6])
        b = np.array([1,2,2,3,3,4,4,5,5,6,6,0,0,1])
        d = np.array([1,2,1,2,1,2,1,2,1,2,1,2,1,2]);
        edges = np.transpose(np.vstack((a,b)))
        G = fg.WeightedGraph(7, edges,d)
        G.symmeterize()
        seed = np.array([0,6])
        label = G.Voronoi_Labelling(seed)
        self.assert_(label[1]==0)

        
    def test_voronoi2(self):
        a = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6])
        b = np.array([1,2,2,3,3,4,4,5,5,6,6,0,0,1])
        d = np.array([1,2,1,2,1,2,1,2,1,2,1,2,1,2]);
        edges = np.transpose(np.vstack((a,b)))
        G = fg.WeightedGraph(7, edges,d)
        G.symmeterize()
        seed = np.array([0])
        label = G.Voronoi_Labelling(seed)
        self.assert_(label[4]==0)
 
    def test_voronoi3(self):
        a = np.array([0,1,2,5,6])
        b = np.array([1,2,3,6,0])
        d = np.array([1,1,1,1,1]);
        edges = np.transpose(np.vstack((a,b)))
        G = fg.WeightedGraph(7, edges,d)
        G.symmeterize()
        seed = np.array([0])
        label = G.Voronoi_Labelling(seed)
        self.assert_(label[4]==-1)

    def test_concatenate1(self,n=10,verbose=0):
        x1 = nr.randn(n,2) 
        x2 = nr.randn(n,2) 
        G1 = fg.WeightedGraph(x1.shape[0])
        G2 = fg.WeightedGraph(x2.shape[0])
        G1.knn(x1,5)
        G2.knn(x2,5) 
        G = fg.concatenate_graphs(G1,G2)
        if verbose:
            G.plot(np.hstack((x1,x2)))
        self.assert_(G.cc().max()>0)

    def test_concatenate2(self,n=10,verbose=0):
        G1 = fg.WeightedGraph(n)
        G2 = fg.WeightedGraph(n)
        G1.complete()
        G2.complete()
        G = fg.concatenate_graphs(G1,G2)
        self.assert_(G.cc().max()==1)

    def test_anti_symmeterize(self,verbose=0):
        n=30
        eps = 1.e-7
        M = (nr.rand(n,n)>0.7).astype('f') 
        C = M-M.T
        G = fg.WeightedGraph(n)
        G.from_adjacency(M)
        G.anti_symmeterize()
        A = G.adjacency()
        self.assert_(np.sum(C-A)**2<eps)

    def test_subgraph_1(self,n=10,verbose=0):
        x = nr.randn(n,2) 
        G = fg.WeightedGraph(x.shape[0])
        valid = np.zeros(n)
        g = G.subgraph(valid)
        self.assert_(g==None)

    def test_subgraph_2(self,n=10,verbose=0):
        x = nr.randn(n,2) 
        G = fg.WeightedGraph(x.shape[0])
        G.knn(x,5)
        valid = np.zeros(n)
        valid[:n/2] = 1
        g = G.subgraph(valid)
        self.assert_(g.edges.max()<n/2)

    def tets_converse(self,n=10):
        G = fg.WeightedGraph(n)
        G.complete()
        c = G.converse_edge()
        eps = ((c-np.ravel(np.reshape(np.arange(n**2),(n,n)).T))**2).sum()
        self.assert_(eps<1.e-7)
        
    

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


