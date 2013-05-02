#!/usr/bin/env python
import numpy as np
import numpy.random as nr

from ..field import (field_from_coo_matrix_and_data,
                     field_from_graph_and_data)
from ..graph import wgraph_from_3d_grid

from nose.tools import assert_true, assert_equal

from numpy.testing import assert_array_equal


def basic_field(nx=10, ny=10, nz=10):
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    data = np.sum(xyz, 1).astype(np.float)
    myfield = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    return myfield


def basic_field_random(nx=10, ny=10, nz=1):
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    data = 0.5 * nr.randn(nx * ny * nz, 1) + np.sum(xyz, 1).astype(np.float)
    myfield = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    return myfield


def basic_field_2(nx=10, ny=10, nz=10):
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    toto = xyz - np.array([5, 5, 5])
    data = np.sum(toto ** 2, 1)
    myfield = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    return myfield


def basic_field_3(nx=10, ny=10, nz=10):
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    toto = xyz - np.array([5, 5, 5])
    data = np.abs(np.sum(toto ** 2, 1) - 11 )
    myfield = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    return myfield


def basic_graph(nx=10, ny=10, nz=10):
    xyz = np.reshape(np.indices((nx, ny, nz)), (3, nx * ny * nz)).T
    data = np.zeros(xyz.shape[0])
    myfield = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 26), data)
    return myfield


def test_type_local_max():
    f = basic_field()
    f.field = f.field.astype(np.float32)
    idx, depth = f.get_local_maxima(th=0)
    assert_array_equal(idx, np.array([999]))


def test_max_1():
    myfield  = basic_field()
    myfield.field[555] = 30
    depth = myfield.local_maxima()
    dep = np.zeros(1000, np.int)
    dep[555] = 5
    dep[999] = 3
    assert_true(sum(np.absolute(dep-depth)) < 1.e-7)


def test_max_2():
    myfield  = basic_field()
    myfield.field[555] = 28
    idx, depth = myfield.get_local_maxima()
    assert_true(len(idx) == 2)
    assert_array_equal(idx, np.array([555, 999]))
    assert_array_equal(depth, np.array([5, 3]))


def test_max_3():
    myfield  = basic_field()
    myfield.field[555] = 27
    idx, depth = myfield.get_local_maxima()
    assert_equal(np.size(idx), 2)
    assert_equal(idx[0], 555)
    assert_equal(idx[1], 999)
    assert_equal(depth[0], 5)
    assert_equal(depth[1], 5)


def test_max_4():
    myfield  = basic_field()
    myfield.field[555] = 28
    idx, depth = myfield.get_local_maxima(0, 27.5)
    assert_equal(np.size(idx), 1)
    assert_equal(idx[0], 555)
    assert_equal(depth[0], 1)


def test_smooth_1():
    G  = basic_graph()
    field = np.zeros((1000,1))
    field[555,0] = 1
    G.set_field(field)
    G.diffusion()
    sfield = G.get_field()
    assert_equal(sfield[555], 0)
    assert_equal(sfield[554], 1)
    assert_true(np.abs(sfield[566] - np.sqrt(2)) < 1.e-7)
    assert_true(np.abs(sfield[446] - np.sqrt(3)) < 1.e-7)


def test_smooth_2():
    G  = basic_graph()
    field = np.zeros((1000, 1))
    field[555, 0] = 1
    G.set_field(field)
    G.diffusion(1)
    sfield = G.get_field()
    assert_equal(sfield[555], 0)
    assert_equal(sfield[554], 1)
    assert_true(np.abs(sfield[566] - np.sqrt(2)) < 1.e-7)
    assert_true(np.abs(sfield[446] - np.sqrt(3)) < 1.e-7)
    

def test_dilation():
    myfield  = basic_field()
    myfield.field[555] = 30
    myfield.field[664] = 0
    myfield.dilation(2)
    assert_true(myfield.field[737] == 30)
    assert_true(myfield.field[0] == 6)
    assert_true(myfield.field[999] == 27)
    assert_true(myfield.field[664] == 30)


def test_dilation2():
    # test equality of cython and python versions
    myfield  = basic_field()
    myfield.field[555] = 30
    myfield.field[664] = 0
    h = myfield.copy()
    h.dilation(2)
    g = myfield.copy()
    g.dilation(2, False)
    assert_array_equal(h.field, g.field)


def test_erosion():
    myfield  = basic_field()
    myfield.field[555] = 30
    myfield.field[664] = 0
    myfield.erosion(2)
    field = myfield.get_field()
    assert_true(field[737] == 11)
    assert_true(field[0] == 0)
    assert_true(field[999] == 21)
    assert_true(field[664] == 0)


def test_opening():
    myfield  = basic_field()
    myfield.field[555] = 30
    myfield.field[664] = 0
    myfield.opening(2)
    field = myfield.get_field()
    assert_true(field[737] == 17)
    assert_true(field[0] == 0)
    assert_true(field[999] == 21)
    assert_true(field[555] == 16)


def test_closing():
    myfield  = basic_field()
    myfield.field[555] = 30
    myfield.field[664] = 0
    myfield.closing(2)
    field = myfield.get_field()
    assert_true(field[737] == 17)
    assert_true(field[0] == 6)
    assert_true(field[999] == 27)
    assert_true(field[555] == 30)


def test_watershed_1():
    myfield = basic_field()
    myfield.field[555] = 28
    myfield.field[664] = 0
    idx, label = myfield.custom_watershed()
    assert_equal(np.size(idx), 2)
    assert_equal(tuple(idx), (555, 999))
    assert_equal((label[776], label[666], label[123]), (1, 0, 0))


def test_watershed_4():
    myfield = basic_field_3()
    idx, label = myfield.custom_watershed()
    assert_true(np.size(idx) == 9)
    assert_true(np.unique(
            [label[555], label[0], label[9], label[90], label[99], label[900],
             label[909], label[990], label[999]]).size == 9)

    
def test_watershed_2():
    myfield = basic_field_2()
    myfield.field[555] = 10
    myfield.field[664] = 0
    idx, label = myfield.custom_watershed()
    assert_true(np.size(idx) == 9)


def test_watershed_3():
    myfield  = basic_field_2()
    myfield.field[555] = 10
    myfield.field[664] = 0
    idx, label = myfield.custom_watershed(0,11)
    assert_true(np.size(idx)==8)


def test_bifurcations_1():
    myfield = basic_field()
    idx, parent,label = myfield.threshold_bifurcations()
    assert_true(idx == 999)
    assert_true(parent == 0)


def test_bifurcations_2():
    myfield = basic_field_2()
    idx, parent, label = myfield.threshold_bifurcations()
    assert_true(np.size(idx) == 15)



def test_geodesic_kmeans(nbseeds=3):
    # Test the geodisc k-means algorithm
    myfield = basic_field_random(5, 5, 1)
    seeds = np.argsort(nr.rand(myfield.V))[:nbseeds]
    seeds, label, inertia = myfield.geodesic_kmeans(seeds)
    assert_array_equal(label[seeds], np.arange(nbseeds))
    assert_true(np.array([i in np.unique(label)
                          for i in np.arange(nbseeds)]).all())


def test_constrained_voronoi(nbseeds=3):
    # Test the geodisc k-means algorithm
    myfield = basic_field_random()
    seeds = np.argsort(nr.rand(myfield.V))[:nbseeds]
    label = myfield.constrained_voronoi(seeds)
    assert_array_equal(label[seeds], np.arange(nbseeds))
    assert_true(np.array([i in np.unique(label)
                          for i in np.arange(nbseeds)]).all())


def test_constrained_voronoi_2(nbseeds=3):
    # Test the geodisc k-means algorithm
    xyz, x = np.zeros((30, 3)), np.arange(30)
    xyz[:, 0] = x
    y = np.array((x // 10), np.float)
    myfield = field_from_graph_and_data(wgraph_from_3d_grid(xyz, 6),  y)
    seeds = np.array([1, 18, 25])
    label = myfield.constrained_voronoi(seeds)
    assert_array_equal(label, x // 10)


def test_subfield():
    myfield = basic_field_random()
    valid = nr.rand(myfield.V) > 0.1
    sf = myfield.subfield(valid)
    assert_equal(sf.V, np.sum(valid))


def test_subfield2():
    myfield = basic_field_random()
    valid = np.zeros(myfield.V)
    sf = myfield.subfield(valid)
    assert_true(sf == None)


def test_ward1():
    myfield = basic_field_random()
    lab, J = myfield.ward(10)
    assert_equal(lab.max(), 9)


def test_ward2():
    myfield = basic_field_random()
    Lab, J1 = myfield.ward(5)
    Lab, J2 = myfield.ward(10)
    assert_true(J1 > J2)


def test_field_from_coo_matrix():
    import scipy.sparse as sps
    V = 10
    a = np.random.rand(V, V) > .9
    fi = field_from_coo_matrix_and_data(sps.coo_matrix(a), a)
    assert_equal(fi.E, a.sum())


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
