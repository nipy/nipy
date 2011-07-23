# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import numpy.linalg as L
import numpy.random as R

from nipy.testing import (assert_equal, assert_almost_equal, dec, parametric)

from nipy.algorithms.statistics import intvol, utils


def symnormal(p=10):
    M = R.standard_normal((p,p))
    return (M + M.T) / np.sqrt(2)


def randorth(p=10):
    """
    A random orthogonal matrix.
    """
    A = symnormal(p)
    return L.eig(A)[1]
                

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
    edges = [R.random_integers(0, shape[j], size=(2,)) 
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
    for v in utils.combinations(range(l), order):
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
    10.0
    >>> intvol.Lips1d(c, b2)
    28.0
    >>> intvol.Lips1d(c, b1+b2)
    39.0

    The function creates two boxes such that
    the 'dilated' box1 does not intersect with box2.
    Additivity works in this case.
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


def test_mu3tet():
    assert_equal(intvol.mu3_tet(0,0,0,0,1,0,0,1,0,1), 1./6)


def test_mu2tri():
    assert_equal(intvol.mu2_tri(0,0,0,1,0,1), 1./2)


def test_mu1tri():
    assert_equal(intvol.mu1_tri(0,0,0,1,0,1), 1+np.sqrt(2)/2)


def test_mu2tet():
    assert_equal(intvol.mu2_tet(0,0,0,0,1,0,0,1,0,1), (3./2 + np.sqrt(3./4))/2)


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
        yield assert_almost_equal, e(box1 + box2), e(box1) + e(box2)


def test_lips1_disjoint():
    phi = intvol.Lips1d
    box1, box2, edge1, edge2 = nonintersecting_boxes((30,))
    c = np.indices((30,)).astype(np.float)
    d = np.random.standard_normal((10,)+(30,))

    U = randorth(p=6)[:1]
    e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
    e.shape = (e.shape[0],) +  c.shape[1:]

    yield assert_almost_equal, phi(c, box1 + box2), \
        phi(c, box1) + phi(c, box2)
    yield assert_almost_equal, phi(d, box1 + box2), \
        phi(d, box1) + phi(d, box2)
    yield assert_almost_equal, phi(e, box1 + box2), \
        phi(e, box1) + phi(e, box2)
    yield assert_almost_equal, phi(e, box1 + box2), phi(c, box1 + box2) 
    yield assert_almost_equal, phi(e, box1 + box2), \
        (np.array([elsym([e[1]-e[0]-1 for e in edge1], i) for i in range(2)])+
         np.array([elsym([e[1]-e[0]-1 for e in edge2], i) for i in range(2)]))


def test_lips2_disjoint():
    phi = intvol.Lips2d
    box1, box2, edge1, edge2 = nonintersecting_boxes((40,40))
    c = np.indices((40,40)).astype(np.float)
    d = np.random.standard_normal((40,40,40))
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


@parametric
def test_lips3_disjoint():
    phi = intvol.Lips3d
    box1, box2, edge1, edge2 = nonintersecting_boxes((40,)*3)
    c = np.indices((40,)*3).astype(np.float)
    d = np.random.standard_normal((40,40,40,40))

    U = randorth(p=6)[0:3]
    e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
    e.shape = (e.shape[0],) +  c.shape[1:]

    yield assert_almost_equal(phi(c, box1 + box2), phi(c, box1) + phi(c, box2))
    yield assert_almost_equal(phi(d, box1 + box2), phi(d, box1) + phi(d, box2))
    yield assert_almost_equal(phi(e, box1 + box2), phi(e, box1) + phi(e, box2))
    yield assert_almost_equal(phi(e, box1 + box2), phi(c, box1 + box2))
    yield assert_almost_equal(
        phi(e, box1 + box2),
        (np.array([elsym([e[1]-e[0]-1 for e in edge1], i) for i in range(4)]) +
         np.array([elsym([e[1]-e[0]-1 for e in edge2], i) for i in range(4)])))


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
        



