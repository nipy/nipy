from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from itertools import combinations

import numpy as np
import numpy.linalg as npl

from .. import intvol

from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_almost_equal


def symnormal(p=10):
    M = np.random.standard_normal((p,p))
    return (M + M.T) / np.sqrt(2)


def randorth(p=10):
    """
    A random orthogonal matrix.
    """
    A = symnormal(p)
    return npl.eig(A)[1]


def box(shape, edges):
    data = np.zeros(shape)
    sl = []
    for i in range(len(shape)):
        sl.append(slice(edges[i][0], edges[i][1],1))
    data[sl] = 1
    return data.astype(np.int)


def randombox(shape):
    """
    Generate a random box, returning the box and the edge lengths
    """
    edges = [np.random.random_integers(0, shape[j], size=(2,))
             for j in range(len(shape))]

    for j in range(len(shape)):
        edges[j].sort()
        if edges[j][0] == edges[j][1]:
            edges[j][0] = 0; edges[j][1] = shape[j]/2+1
    return edges, box(shape, edges)


def elsym(edgelen, order=1):
    """
    Elementary symmetric polynomial of a given order
    """
    l = len(edgelen)
    if order == 0:
        return 1
    r = 0
    for v in combinations(range(l), order):
        r += np.product([edgelen[vv] for vv in v])
    return r


def nonintersecting_boxes(shape):
    """
    The Lips's are supposed to be additive, so disjoint things
    should be additive. But, if they ALMOST intersect, different
    things get added to the triangulation.

    >>> b1 = np.zeros(40, np.int)
    >>> b1[:11] = 1
    >>> b2 = np.zeros(40, np.int)
    >>> b2[11:] = 1
    >>> (b1*b2).sum()
    0
    >>> c = np.indices((40,)).astype(np.float)
    >>> intvol.Lips1d(c, b1)
    array([  1.,  10.])
    >>> intvol.Lips1d(c, b2)
    array([  1.,  28.])
    >>> intvol.Lips1d(c, b1+b2)
    array([  1.,  39.])

    The function creates two boxes such that the 'dilated' box1 does not
    intersect with box2.  Additivity works in this case.
    """
    while True:
        edge1, box1 = randombox(shape)
        edge2, box2 = randombox(shape)

        diledge1 = [[max(ed[0]-1, 0), min(ed[1]+1, sh)] 
                    for ed, sh in zip(edge1, box1.shape)]

        dilbox1 = box(box1.shape, diledge1)

        if set(np.unique(dilbox1 + box2)).issubset([0,1]):
            break
    return box1, box2, edge1, edge2


def pts2dots(d, a, b, c):
    """ Convert point coordinates to dot products
    """
    D00 = np.dot(d, d)
    D01 = np.dot(d, a)
    D02 = np.dot(d, b)
    D03 = np.dot(d, c)
    D11 = np.dot(a, a)
    D12 = np.dot(a, b)
    D13 = np.dot(a, c)
    D22 = np.dot(b, b)
    D23 = np.dot(b, c)
    D33 = np.dot(c, c)
    return D00, D01, D02, D03, D11, D12, D13, D22, D23, D33


def pts2mu3_tet(d, a, b, c):
    """ Accept point coordinates for calling mu3tet
    """
    return intvol.mu3_tet(*pts2dots(d, a, b, c))


def wiki_tet_vol(d, a, b, c):
    # Wikipedia formula for generalized tetrahedron volume
    d, a, b, c = [np.array(e) for e in (d, a, b, c)]
    cp = np.cross((b-d),(c-d))
    v2t6 = np.dot((a-d), cp)
    return np.sqrt(v2t6) / 6.


def test_mu3tet():
    assert_equal(intvol.mu3_tet(0,0,0,0,1,0,0,1,0,1), 1./6)
    assert_equal(intvol.mu3_tet(0,0,0,0,0,0,0,0,0,0), 0)
    d = [2,2,2]
    a = [3,2,2]
    b = [2,3,2]
    c = [2,2,3]
    assert_equal(pts2mu3_tet(d, a, b, c), 1./6)
    assert_equal(wiki_tet_vol(d, a, b, c), 1./6)
    # This used to generate nan values
    assert_equal(intvol.mu3_tet(0,0,0,0,1,0,0,-1,0,1), 0)


def test_mu2tri():
    assert_equal(intvol.mu2_tri(0,0,0,1,0,1), 1./2)


def test_mu1tri():
    assert_equal(intvol.mu1_tri(0,0,0,1,0,1), 1+np.sqrt(2)/2)


def test_mu2tet():
    assert_equal(intvol.mu2_tet(0,0,0,0,1,0,0,1,0,1), (3./2 + np.sqrt(3./4))/2)


def pts2mu1_tet(d, a, b, c):
    """ Accept point coordinates for calling mu1_tet
    """
    return intvol.mu1_tet(*pts2dots(d, a, b, c))


def test_mu1_tet():
    res1 = pts2mu1_tet([2,2,2],[3,2,2],[2,3,2],[2,2,3])
    res2 = pts2mu1_tet([0,0,0],[1,0,0],[0,1,0],[0,0,1])
    assert_equal(res1, res2)
    assert_equal(intvol.mu1_tet(0,0,0,0,0,0,0,0,0,0), 0)
    # This used to generate nan values
    assert_equal(intvol.mu1_tet(0,0,0,0,1,0,0,-1,0,1), 0)


def test__mu1_tetface():
    # Test for out of range acos value sequences.  I'm ashamed to say I found
    # these sequences accidentally in a failing test with random numbers
    _mu1_tetface = intvol._mu1_tetface
    assert_almost_equal(_mu1_tetface(1, 0, 0, 10, 10, 0, 0, 20, 20, 40), 0)
    assert_almost_equal(_mu1_tetface(36, 0, 0, 18, 48, 0, 0, 1, 30,  63), 3)


def test_ec():
    for i in range(1, 4):
        _, box1 = randombox((40,)*i)
        f = {3:intvol.EC3d,
             2:intvol.EC2d,
             1:intvol.EC1d}[i]
        yield assert_almost_equal, f(box1), 1


def test_ec_disjoint():
    for i in range(1, 4):
        e = {3:intvol.EC3d,
             2:intvol.EC2d,
             1:intvol.EC1d}[i]
        box1, box2, _, _ = nonintersecting_boxes((40,)*i)
        assert_almost_equal(e(box1 + box2), e(box1) + e(box2))


def test_lips_wrapping():
    # Test that shapes touching the edge do not combine by wrapping
    b1 = np.zeros(40, np.int)
    b1[:11] = 1
    b2 = np.zeros(40, np.int)
    b2[11:] = 1
    # lines are disjoint
    assert_equal((b1*b2).sum(), 0)
    c = np.indices(b1.shape).astype(np.float)
    assert_array_equal(intvol.Lips1d(c, b1), (1, 10))
    assert_array_equal(intvol.Lips1d(c, b2), (1, 28))
    assert_array_equal(intvol.Lips1d(c, b1+b2), (1, 39.0))
    # 2D
    b1 = b1[:,None]
    b2 = b2[:,None]
    # boxes are disjoint
    assert_equal((b1*b2).sum(), 0)
    c = np.indices(b1.shape).astype(np.float)
    assert_array_equal(intvol.Lips2d(c, b1), (1, 10, 0))
    assert_array_equal(intvol.Lips2d(c, b2), (1, 28, 0))
    assert_array_equal(intvol.Lips2d(c, b1+b2), (1, 39.0, 0))
    # 3D
    b1 = b1[:,:,None]
    b2 = b2[:,:,None]
    assert_equal(b1.shape, (40,1,1))
    # boxes are disjoint
    assert_equal((b1*b2).sum(), 0)
    c = np.indices(b1.shape).astype(np.float)
    assert_array_equal(intvol.Lips3d(c, b1), (1, 10, 0, 0))
    assert_array_equal(intvol.Lips3d(c, b2), (1, 28, 0, 0))
    assert_array_equal(intvol.Lips3d(c, b1+b2), (1, 39.0, 0, 0))
    # Shapes which are squeezable should still return sensible answers
    # Test simple ones line / box / volume
    funcer = {1: (intvol.Lips1d, intvol.EC1d),
              2: (intvol.Lips2d, intvol.EC2d),
              3: (intvol.Lips3d, intvol.EC3d)}
    for box_shape, exp_ivs in [[(10,),(1,9)],
                               [(10,1),(1,9,0)],
                               [(1,10),(1,9,0)],
                               [(10,1,1), (1,9,0,0)],
                               [(1, 10, 1), (1,9,0,0)],
                               [(1, 1, 10), (1,9,0,0)]]:
        nd = len(box_shape)
        lips_func, ec_func = funcer[nd]
        c = np.indices(box_shape).astype(np.float)
        b = np.ones(box_shape, dtype=np.int)
        assert_array_equal(lips_func(c, b), exp_ivs)
        assert_equal(ec_func(b), exp_ivs[0])


def test_lips1_disjoint():
    phi = intvol.Lips1d
    box1, box2, edge1, edge2 = nonintersecting_boxes((30,))
    c = np.indices((30,)).astype(np.float)
    # Test N dimensional coordinates (N=10)
    d = np.random.standard_normal((10,)+(30,))
    # Test rotation causes no change in volumes
    U = randorth(p=6)[:1]
    e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
    e.shape = (e.shape[0],) +  c.shape[1:]

    assert_almost_equal(phi(c, box1 + box2), phi(c, box1) + phi(c, box2))
    assert_almost_equal(phi(d, box1 + box2), phi(d, box1) + phi(d, box2))
    assert_almost_equal(phi(e, box1 + box2), phi(e, box1) + phi(e, box2))
    assert_almost_equal(phi(e, box1 + box2), phi(c, box1 + box2))
    assert_almost_equal(phi(e, box1 + box2),
                        (np.array(
                            [elsym([e[1]-e[0]-1
                                    for e in edge1], i) for i in range(2)]) +
                        np.array(
                            [elsym([e[1]-e[0]-1
                                    for e in edge2], i) for i in range(2)])))
    assert_raises(ValueError, phi, c[...,None], box1)


def test_lips2_disjoint():
    phi = intvol.Lips2d
    box1, box2, edge1, edge2 = nonintersecting_boxes((40,40))
    c = np.indices((40,40)).astype(np.float)
    # Test N dimensional coordinates (N=10)
    d = np.random.standard_normal((10,40,40))
    # Test rotation causes no change in volumes
    U = randorth(p=6)[0:2]
    e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
    e.shape = (e.shape[0],) +  c.shape[1:]
    assert_almost_equal(phi(c, box1 + box2),
                        phi(c, box1) + phi(c, box2))
    assert_almost_equal(phi(d, box1 + box2),
                        phi(d, box1) + phi(d, box2))
    assert_almost_equal(phi(e, box1 + box2),
                        phi(e, box1) + phi(e, box2))
    assert_almost_equal(phi(e, box1 + box2), phi(c, box1 + box2))
    assert_almost_equal(phi(e, box1 + box2),
                        np.array([elsym([e[1]-e[0]-1 for e in edge1], i)
                                   for i in range(3)]) +
                        np.array([elsym([e[1]-e[0]-1 for e in edge2], i)
                                  for i in range(3)])
                       )
    assert_raises(ValueError, phi, c[...,None], box1)
    assert_raises(ValueError, phi, c[:,:,1], box1)


def test_lips3_disjoint():
    phi = intvol.Lips3d
    box1, box2, edge1, edge2 = nonintersecting_boxes((40,)*3)
    c = np.indices((40,)*3).astype(np.float)
    # Test N dimensional coordinates (N=10)
    d = np.random.standard_normal((10,40,40,40))
    # Test rotation causes no change in volumes
    U = randorth(p=6)[0:3]
    e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
    e.shape = (e.shape[0],) +  c.shape[1:]

    assert_almost_equal(phi(c, box1 + box2), phi(c, box1) + phi(c, box2))
    assert_almost_equal(phi(d, box1 + box2), phi(d, box1) + phi(d, box2))
    assert_almost_equal(phi(e, box1 + box2), phi(e, box1) + phi(e, box2))
    assert_almost_equal(phi(e, box1 + box2), phi(c, box1 + box2))
    assert_almost_equal(
        phi(e, box1 + box2),
        (np.array([elsym([e[1]-e[0]-1 for e in edge1], i) for i in range(4)]) +
         np.array([elsym([e[1]-e[0]-1 for e in edge2], i) for i in range(4)])))
    assert_raises(ValueError, phi, c[...,None], box1)
    assert_raises(ValueError, phi, c[:,:,:,1], box1)


def test_lips3_nans():
    # These boxes caused nans in the Lips3 disjoint box tests
    phi = intvol.Lips3d
    box1 = np.zeros((40,40,40), dtype=np.int)
    box2 = box1.copy()
    box1[23:30,22:32,9:13] = 1
    box2[7:22,0,8:17] = 1
    c = np.indices(box1.shape).astype(np.float)
    assert_array_equal(np.isnan(phi(c, box2)), False)
    U = randorth(p=6)[0:3]
    e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
    e.shape = (e.shape[0],) +  c.shape[1:]
    assert_array_equal(np.isnan(phi(e, box1 + box2)), False)


def test_slices():
    # Slices have EC 1...
    e = intvol.EC3d
    p = intvol.Lips3d
    m = np.zeros((40,)*3, np.int)
    D = np.indices(m.shape).astype(np.float)
    m[10,10,10] = 1
    yield assert_almost_equal, e(m), 1
    yield assert_almost_equal, p(D,m), [1,0,0,0]

    m = np.zeros((40,)*3, np.int)
    m[10,10:14,10] = 1
    yield assert_almost_equal, e(m), 1
    yield assert_almost_equal, p(D,m), [1,3,0,0]

    m = np.zeros((40,)*3, np.int)
    m[10,10:14,9:15] = 1
    yield assert_almost_equal, e(m), 1
    yield assert_almost_equal, p(D,m), [1,8,15,0]


def test_ec_wrapping():
    # Test wrapping for EC1 calculation
    assert_equal(intvol.EC1d(np.ones((6,), dtype=np.int)), 1)
    box1 = np.array([1, 1, 0, 1, 1, 1], dtype=np.int)
    assert_equal(intvol.EC1d(box1), 2)
    # 2D
    box1 = np.zeros((3,6), dtype=np.int)
    box1[1] = 1
    assert_equal(intvol.EC2d(box1), 1)
    box1[1, 3] = 0
    assert_equal(intvol.EC2d(box1), 2)
    # 3D
    box1 = np.zeros((3,6,3), dtype=np.int)
    box1[1, :, 1] = 1
    assert_equal(intvol.EC3d(box1), 1)
    box1[1, 3, 1] = 0
    assert_equal(intvol.EC3d(box1), 2)
