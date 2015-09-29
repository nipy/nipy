from __future__ import absolute_import
#!/usr/bin/env python


#
# Test BLAS 3
#


from numpy.testing import assert_almost_equal
import numpy as np
from .. import (blas_dgemm, blas_dsymm, blas_dtrmm, 
                blas_dtrsm, blas_dsyrk, blas_dsyr2k)

n1 = 10
n2 = 13

    
def test_dgemm():
    A = np.random.rand(n1,n2)
    B = np.random.rand(n2,n1)
    C = np.random.rand(n1,n1)
    C2 = np.random.rand(n2,n2)
    alpha = np.double(np.random.rand(1))
    beta = np.double(np.random.rand(1))
    # Test: A*B
    Dgold = alpha*np.dot(A,B) + beta*C
    D = blas_dgemm(0, 0, alpha, A, B, beta, C)
    assert_almost_equal(Dgold, D) 
    # Test: A^t B^t
    Dgold = alpha*np.dot(A.T,B.T) + beta*C2
    D = blas_dgemm(1, 1, alpha, A, B, beta, C2)
    assert_almost_equal(Dgold, D) 

def test_dsymm():
    A = np.random.rand(n1,n1)
    A = A + A.T
    B = np.random.rand(n1,n2)
    C = np.random.rand(n1,n2)
    B2 = np.random.rand(n2,n1)
    C2 = np.random.rand(n2,n1)
    alpha = np.double(np.random.rand(1))
    beta = np.double(np.random.rand(1))
    # Test: A*B
    Dgold = alpha*np.dot(A,B) + beta*C
    D = blas_dsymm(0, 0, alpha, A, B, beta, C)
    assert_almost_equal(Dgold, D)
    D = blas_dsymm(0, 1, alpha, A, B, beta, C)
    assert_almost_equal(Dgold, D)
    # Test: B*A
    Dgold = alpha*np.dot(B2,A) + beta*C2
    D = blas_dsymm(1, 0, alpha, A, B2, beta, C2)
    assert_almost_equal(Dgold, D)
    D = blas_dsymm(1, 1, alpha, A, B2, beta, C2)
    assert_almost_equal(Dgold, D)

def _test_dtrXm(A, U, L, B, alpha, blasfn):
    # Test: U*B
    Dgold = alpha*np.dot(U,B) 
    D = blasfn(0, 0, 0, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: B*U
    Dgold = alpha*np.dot(B,U)
    D = blasfn(1, 0, 0, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: U'*B
    Dgold = alpha*np.dot(U.T,B) 
    D = blasfn(0, 0, 1, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: B*U'
    Dgold = alpha*np.dot(B,U.T)
    D = blasfn(1, 0, 1, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: L*B
    Dgold = alpha*np.dot(L,B) 
    D = blasfn(0, 1, 0, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: B*L
    Dgold = alpha*np.dot(B,L)
    D = blasfn(1, 1, 0, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: L'*B
    Dgold = alpha*np.dot(L.T,B) 
    D = blasfn(0, 1, 1, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: B*L'
    Dgold = alpha*np.dot(B,L.T)
    D = blasfn(1, 1, 1, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: U*B
    Dgold = alpha*np.dot(U,B) 
    D = blasfn(0, 0, 0, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: B*U
    Dgold = alpha*np.dot(B,U)
    D = blasfn(1, 0, 0, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: U'*B
    Dgold = alpha*np.dot(U.T,B) 
    D = blasfn(0, 0, 1, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: B*U'
    Dgold = alpha*np.dot(B,U.T)
    D = blasfn(1, 0, 1, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: L*B
    Dgold = alpha*np.dot(L,B) 
    D = blasfn(0, 1, 0, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: B*L
    Dgold = alpha*np.dot(B,L)
    D = blasfn(1, 1, 0, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: L'*B
    Dgold = alpha*np.dot(L.T,B) 
    D = blasfn(0, 1, 1, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)
    # Test: B*L'
    Dgold = alpha*np.dot(B,L.T)
    D = blasfn(1, 1, 1, 0, alpha, A, B)
    assert_almost_equal(Dgold, D)

def test_dtrmm():
    A = np.random.rand(n1,n1)
    U = np.triu(A)
    L = np.tril(A)
    B = np.random.rand(n1,n1)
    alpha = np.double(np.random.rand(1))
    _test_dtrXm(A, U, L, B, alpha, blas_dtrmm)

def test_dtrsm():
    A = np.random.rand(n1,n1)
    U = np.linalg.inv(np.triu(A))
    L = np.linalg.inv(np.tril(A))
    B = np.random.rand(n1,n1)
    alpha = np.double(np.random.rand(1))
    _test_dtrXm(A, U, L, B, alpha, blas_dtrsm)
    
def test_dsyrk(): 
    A = np.random.rand(n1,n1)
    C = np.random.rand(n1,n1)
    alpha = np.double(np.random.rand(1))
    beta = np.double(np.random.rand(1))
    # Test A*A'
    U = np.triu(blas_dsyrk(0, 0, alpha, A, beta, C))
    L = np.tril(blas_dsyrk(1, 0, alpha, A, beta, C))
    Dgold = alpha*np.dot(A, A.T) + beta*C
    Ugold = np.triu(Dgold)
    Lgold = np.tril(Dgold)
    assert_almost_equal(Ugold, U)
    assert_almost_equal(Lgold, L)
    # Test A'*A
    U = np.triu(blas_dsyrk(0, 1, alpha, A, beta, C))
    L = np.tril(blas_dsyrk(1, 1, alpha, A, beta, C))
    Dgold = alpha*np.dot(A.T, A) + beta*C
    Ugold = np.triu(Dgold)
    Lgold = np.tril(Dgold)
    assert_almost_equal(Ugold, U)
    assert_almost_equal(Lgold, L)

def test_dsyr2k(): 
    A = np.random.rand(n1,n1)
    B = np.random.rand(n1,n1)
    C = np.random.rand(n1,n1)
    alpha = np.double(np.random.rand(1))
    beta = np.double(np.random.rand(1))
    # Test A*B' + B*A'
    U = np.triu(blas_dsyr2k(0, 0, alpha, A, B, beta, C))
    L = np.tril(blas_dsyr2k(1, 0, alpha, A, B, beta, C))
    Dgold = alpha*(np.dot(A,B.T) + np.dot(B,A.T)) + beta*C
    Ugold = np.triu(Dgold)
    Lgold = np.tril(Dgold)
    assert_almost_equal(Ugold, U)
    assert_almost_equal(Lgold, L)
    # Test A'*B + B'*A
    U = np.triu(blas_dsyr2k(0, 1, alpha, A, B, beta, C))
    L = np.tril(blas_dsyr2k(1, 1, alpha, A, B, beta, C))
    Dgold = alpha*(np.dot(A.T,B) + np.dot(B.T,A)) + beta*C
    Ugold = np.triu(Dgold)
    Lgold = np.tril(Dgold)
    assert_almost_equal(Ugold, U)
    assert_almost_equal(Lgold, L)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

