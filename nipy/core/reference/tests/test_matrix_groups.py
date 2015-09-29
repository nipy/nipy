from __future__ import absolute_import

import numpy as np
from scipy.linalg import expm

from nose.tools import assert_true, assert_equal, assert_raises

import nipy.core.reference.tests.matrix_groups as MG
from nipy.core.api import ArrayCoordMap

A = np.array([[0,1],
              [1,0]])

B = np.array([[5,4],
              [4,3]]) 

D = np.array([[25,4],
              [31,5]])

def test_init():
    """
    Test that we can initialize the MatrixGroup subclasses
    """
    O_A = MG.O(A, 'xy')
    GLR_A = MG.GLR(A, 'xy')
    GLZ_A = MG.GLZ(A, 'xy')
    SO_A = MG.SO(np.identity(2), 'xy')

    B = np.array([[np.sin(0.75), np.cos(0.75)],
                  [-np.cos(0.75), np.sin(0.75)]])
    O_B = MG.O(B, 'xy')
    SO_B = MG.O(B, 'xy')

    B[1] = -B[1]
    assert_raises(ValueError, MG.SO, B, 'xy')
    O_B = MG.O(B, 'xy')

def test2():
    Z = np.random.standard_normal((3,3))
    GL_Z = MG.GLR(Z, 'xyz')

    assert_raises(ValueError, MG.SO, Z, 'zxy')

    detZ = np.linalg.det(Z)
    if detZ < 0:
        W = -Z
    else: W = Z
    f = np.fabs(detZ)**(1/3.)
    SL_Z = MG.SLR(W/f, 'xyz')

    orth = expm(Z - Z.T)
    O_Z = MG.O(orth, 'xyz')

def random_orth(dim=3, names=None):
    Z = np.random.standard_normal((3,3))
    orth = expm(Z - Z.T)
    if not names:
        names = ['e%d' % i for i in range(dim)]
    else:
        if len(names) != dim:
            raise ValueError('len(names) != dim')
    return MG.O(orth, names)

def test_basis_change():

    basis1 = random_orth(names='xyz')
    basis2 = random_orth(names='uvw')
    
    bchange = MG.Linear(basis2.coords, basis1.coords, random_orth(dim=3).matrix)
    #print basis1.coords
    new = MG.change_basis(basis1, bchange)

    assert_true(MG.same_transformation(basis1, new, bchange))

def test_product():

    GLZ_A = MG.GLZ(A, 'xy')
    GLZ_B = MG.GLZ(B, 'xy')
    GLZ_C = MG.GLZ(B, 'ij')

    GLZ_AB = MG.product(GLZ_A, GLZ_B)
    yield (assert_true, 
           np.allclose(GLZ_AB.matrix, np.dot(GLZ_A.matrix, GLZ_B.matrix)))

    # different coordinates: can't make the product
    yield assert_raises, ValueError, MG.product, GLZ_A, GLZ_C

def test_product2():
    O_1 = random_orth(names='xyz')
    O_2 = random_orth(names='xyz')
    O_21 = MG.product(O_2, O_1)
    #print type(O_21)

def test_homomorphism():

    GLZ_B = MG.GLZ(B, 'xy')
    GLZ_D = MG.GLZ(D, 'ij')
    GLZ_BD = MG.product_homomorphism(GLZ_B, GLZ_D)

    yield assert_true, np.allclose(GLZ_BD.matrix[:2,:2], GLZ_B.matrix)
    yield assert_true, np.allclose(GLZ_BD.matrix[2:,2:], GLZ_D.matrix)
    yield assert_true, np.allclose(GLZ_BD.matrix[2:,:2], 0)
    yield assert_true, np.allclose(GLZ_BD.matrix[:2,2:], 0)

    GLZ_C = MG.GLZ(D, 'xy')
    # have the same axisnames, an exception will be raised
    yield assert_raises, ValueError, MG.product_homomorphism, GLZ_C, GLZ_B

    E = np.array([[7,8],
                  [8,9]])
    GLZ_E = MG.GLZ(E, 'ij')

    F = np.array([[6,7],
                  [5,6]])
    GLZ_F = MG.GLZ(E, 'xy')

    GLZ_FE = MG.product_homomorphism(GLZ_F, GLZ_E)

    test1 = MG.product(GLZ_FE, GLZ_BD)
    test2 = MG.product_homomorphism(MG.product(GLZ_F, GLZ_B), MG.product(GLZ_E, GLZ_D))
    yield assert_true, np.allclose(test1.matrix, test2.matrix)


def test_32():
    class O32(MG.O):
        dtype = np.float32

        def validate(self, M=None):
            """
            Check that the matrix is (almost) orthogonal.
            """
            if M is None:
                M = self.matrix
            return np.allclose(np.identity(self.ndims[0], dtype=self.dtype), np.dot(M.T, M), atol=1.0e-06)

    a = random_orth(3).matrix.astype(np.float32)

    A = O32(a, 'xyz')
    B = O32(random_orth(3).matrix.astype(np.float32), 'xyz')
    C = MG.product(A, B)
    yield assert_equal, C.dtype, np.float32

    ev = ArrayCoordMap.from_shape(C, (20,30,40))
    yield assert_equal, ev.values.dtype, np.float32
