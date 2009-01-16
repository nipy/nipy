import numpy as N
from numpy.linalg import svd

from reml import orth

def _trace(x):
    """
    Trace of a square 2d array.

    Does not check
    shape of x to ensure it's square.
    """
    return N.diag(x).sum()

def _frobenius(A, B):
    """
    Frobenius inner product of A and B: Trace(A'B)

    Does not check
    shape of x to ensure it's square.
    """
    return (A * B).sum()

def trRV(X=None, V=None):
    """
    If V is None it defaults to identity.

    If X is None, it defaults to the 0-dimensional subspace,
    i.e. R is the identity.

    >>> from numpy import *
    >>> from numpy.random import standard_normal
    >>>
    >>> X = standard_normal((100, 4))
    >>> print trRV(X)
    (96.0, 96.0)
    >>>
    >>> V = identity(100)
    >>> print trRV(X, V)
    (96.0, 96.0)
    >>>
    >>> X[:,3] = X[:,1] + X[:,2]
    >>> print trRV(X, V)
    (97.0, 97.0)
    >>>
    >>> u = orth(X)
    >>> V = dot(u, u.T)
    >>> print N.allclose(trRV(X, V), 0)
    True
    >>>

    """

    n, p = X.shape

    if V is None:
        V = N.identity(n)
        
    if X is None:
        if V is None:
            trRV = trRVRV = n
        else:
            trRV = _trace(V)
            trRVRV = _frobenius(V, V)
    else:
        u = orth(X)
        if V is None:
            trRV = trRVRV = n - u.shape[1]
        else:
            Vu = N.dot(V, u)
            utVu = N.dot(u.T, Vu)
            trRV = _trace(V) - _frobenius(u, Vu)
            trRVRV = _frobenius(V, V) - 2 * _frobenius(Vu, Vu) + _frobenius(utVu, utVu)

    return trRV, trRVRV

if __name__ == "__main__":

    from numpy.random import standard_normal

    X = standard_normal((100, 4))
    print trRV(X) # should be (96,96)

    V = N.identity(100)
    print trRV(X, V) # should be (96,96)

    X[:,3] = X[:,1] + X[:,2]
    print trRV(X, V) # should be (97,97)

    u = orth(X)
    V = N.dot(u, u.T)
    print trRV(X, V) # should be (0,0)
            
