#!/usr/bin/env python
from numpy.testing import TestCase
import numpy as np
from ..field import (Field, field_from_coo_matrix_and_data, 
                     field_from_graph_and_data)
from ..graph import wgraph_from_3d_grid


def basic_field(nx=10, ny=10, nz=10):
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    data = np.sum(xyz, 1).astype(np.float)
    F = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    return F


def basic_field_random(nx=10, ny=10, nz=1):
    import numpy.random as nr
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    data = 0.5 * nr.randn(nx * ny * nz, 1) + np.sum(xyz, 1).astype(np.float)
    F = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    return F

def basic_field_2(nx=10, ny=10, nz=10):
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    toto = xyz - np.array([5, 5, 5])
    data = np.sum(toto ** 2, 1) 
    F = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    return F

def basic_field_3(nx=10, ny=10, nz=10):
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    toto = xyz - np.array([5, 5, 5])
    data = np.abs(np.sum(toto ** 2, 1) - 11 )
    F = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    return F

def basic_graph(nx=10, ny=10, nz=10):
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    data = np.zeros(xyz.shape[0])
    F = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)    
    return F 



class test_Field(TestCase):

    def test_max_1(self):
        F  = basic_field()
        F.field[555] = 30
        depth = F.local_maxima()
        dep = np.zeros(1000, np.int)
        dep[555] = 5 
        dep[999] = 3 
        assert sum(np.absolute(dep-depth))<1.e-7

    def test_max_2(self):
        F  = basic_field()
        F.field[555] = 28
        idx,depth = F.get_local_maxima()
        self.assert_(len(idx) == 2)
        self.assert_(np.alltrue( idx == (555, 999)))
        self.assert_(np.alltrue( depth == (5, 3)))

    def test_max_3(self):
        F  = basic_field()
        F.field[555] = 27
        idx, depth = F.get_local_maxima()
        assert (np.size(idx) == 2) 
        assert (idx[0] == 555)
        assert (idx[1] == 999)
        assert (depth[0] == 5)
        assert (depth[1] == 5)

    def test_max_4(self):
        F  = basic_field()
        F.field[555] = 28
        idx, depth = F.get_local_maxima(0, 27.5)
        assert (np.size(idx) == 1)
        assert (idx[0] == 555)
        assert (depth[0] == 1)

    def test_smooth_1(self):
        G  = basic_graph()
        field = np.zeros((1000,1))
        field[555,0] = 1
        G.set_field(field)
        G.diffusion()
        sfield = G.get_field()
        OK1 = (sfield[555]==0)
        OK2 = (sfield[554]==1)
        OK3 = (np.absolute(sfield[566]-np.sqrt(2))<1.e-7)
        OK4 = (np.absolute(sfield[446]-np.sqrt(3))<1.e-7)
        OK = OK1 & OK2 & OK3 & OK4
        self.assert_(OK)

    def test_smooth_2(self):
        G  = basic_graph()
        field = np.zeros((1000,1))
        field[555,0] = 1
        G.set_field(field)
        G.diffusion(1)
        sfield = G.get_field()
        OK1 = (sfield[555]==0)
        OK2 = (sfield[554]==1)
        OK3 = (np.absolute(sfield[566]-np.sqrt(2))<1.e-7)
        OK4 = (np.absolute(sfield[446]-np.sqrt(3))<1.e-7)
        OK = OK1 & OK2 & OK3 & OK4
        self.assert_(OK)

    def test_dilation(self):
        F  = basic_field()
        F.field[555] = 30
        F.field[664] = 0
        F.dilation(2)
        assert (F.field[737] == 30)
        assert (F.field[0] == 6)
        assert (F.field[999] == 27)
        assert (F.field[664] == 30)

    def test_dilation2(self):
        """ test equality of cuthon and python versions
        """
        F  = basic_field()
        F.field[555] = 30
        F.field[664] = 0
        h = F.copy()
        h.dilation(2)
        g = F.copy()
        g.dilation(2, False)
        assert (h.field == g.field).all()

    def test_erosion(self):
        F  = basic_field()
        F.field[555] = 30
        F.field[664] = 0
        F.erosion(2)
        field = F.get_field()
        assert (field[737] == 11)
        assert (field[0] == 0)
        assert (field[999] == 21)
        assert (field[664] == 0)

    def test_opening(self):
        F  = basic_field()
        F.field[555] = 30
        F.field[664] = 0
        F.opening(2)
        field = F.get_field()
        assert (field[737] == 17)
        assert (field[0] == 0)
        assert (field[999] == 21)
        assert (field[555] == 16)
    
    def test_closing(self):
        F  = basic_field()
        F.field[555] = 30
        F.field[664] = 0
        F.closing(2)
        field = F.get_field()
        assert (field[737] == 17)
        assert (field[0] == 6)
        assert (field[999] == 27)
        assert (field[555] == 30)


    def test_watershed_1(self):
        F = basic_field()
        F.field[555] = 28
        F.field[664] = 0
        idx, label = F.custom_watershed()
        assert np.size(idx) == 2
        assert (idx[0]==555) & (idx[1]==999)
        assert (label[776]==1) & (label[666]==0) & (label[123]==0)
        
    def test_watershed_4(self):
        F = basic_field_3()
        idx, label = F.custom_watershed()
        assert np.size(idx) == 9
        assert np.unique([label[555], label[0], label[9], label[90], 
                          label[99], label[900], label[909], label[990], 
                          label[999]]).size == 9
        
    def test_watershed_2(self):
        F = basic_field_2()
        F.field[555] = 10
        F.field[664] = 0
        idx, label = F.custom_watershed()
        assert np.size(idx)==9

    def test_watershed_3(self):
        F  = basic_field_2()
        F.field[555] = 10
        F.field[664] = 0
        idx, label = F.custom_watershed(0,11)
        assert np.size(idx)==8

    def test_bifurcations_1(self):
        F = basic_field()   
        idx, parent,label = F.threshold_bifurcations()
        assert (idx == 999)
        assert (parent == 0);
        
    def test_bifurcations_2(self):
        F = basic_field_2()
        idx, parent, label = F.threshold_bifurcations()
        assert np.size(idx) == 15

    def test_geodesic_kmeans(self, nbseeds=3):
        """ Test the geodisc k-means algorithm
        """
        import numpy.random as nr
        F = basic_field_random(5, 5, 1)
        seeds = np.argsort(nr.rand(F.V))[:nbseeds]
        seeds, label, inertia = F.geodesic_kmeans(seeds)
        for i in range(nbseeds):
            assert label[seeds[i]] == i     
        print np.unique(label), np.arange(nbseeds)
        assert np.array([i in np.unique(label) 
                         for i in np.arange(nbseeds)]).all()
    
    def test_constrained_voronoi(self, nbseeds=3):
        """ Test the geodisc k-means algorithm
        """
        import numpy.random as nr
        F = basic_field_random()
        seeds = np.argsort(nr.rand(F.V))[:nbseeds]
        label = F.constrained_voronoi(seeds)
        for i in range(nbseeds):
            assert label[seeds[i]] == i
        assert np.array([i in np.unique(label) 
                         for i in np.arange(nbseeds)]).all()

    def test_constrained_voronoi_2(self, nbseeds=3):
        """ Test the geodisc k-means algorithm
        """
        xyz, x = np.zeros((30, 3)), np.arange(30)
        xyz[:, 0] = x
        y = np.array((x / 10), np.float)
        F = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 6),  y)
        seeds = np.array([1, 18, 25])
        label = F.constrained_voronoi(seeds)
        print label
        assert (label == (x / 10)).all()

    def test_subfield(self):
        import numpy.random as nr
        F = basic_field_random()
        valid = nr.rand(F.V)>0.1
        sf = F.subfield(valid)
        self.assert_(sf.V==np.sum(valid))

    def test_subfield2(self):
        F = basic_field_random()
        valid = np.zeros(F.V)
        sf = F.subfield(valid)
        self.assert_(sf==None)

    def test_ward1(self):
        F = basic_field_random()
        Lab, J = F.ward(10)
        self.assert_(Lab.max()==9)

    def test_ward2(self):
        F = basic_field_random()
        Lab, J1 = F.ward(5)
        Lab, J2 = F.ward(10)
        self.assert_(J1>J2)

    def test_field_from_coo_matrix(self):
        import scipy.sparse as sps
        V = 10
        a = np.random.rand(V, V)>.9
        fi = field_from_coo_matrix_and_data(sps.coo_matrix(a), a)
        print fi.E , a.sum()
        self.assert_(fi.E==a.sum())

def test_bifurcation_parents():
    # test that the parents inferred in bifurcation are right
    from numpy.testing import assert_array_equal
    shape = (5, 5, 1)
    xyz = np.reshape(np.indices(shape), (3, np.prod(shape))).T
    toto = xyz - np.array([2, 2, 0])
    data = np.abs(np.sum(toto ** 2, 1) - 2 )
    F = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    idx, parent, label = F.threshold_bifurcations()
    assert_array_equal(parent, np.array([4, 5, 4, 5, 6, 6, 8, 8, 8]))

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


