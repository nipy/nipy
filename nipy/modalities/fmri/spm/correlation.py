import numpy as N
from numpy.linalg import inv

def ARcovariance(rho, n, cor=False, sigma=1.):
    """
    Return covariance matrix of a sample of length n from an AR(p)
    process with parameters rho.

    INPUTS:

    rho      -- an array of length p
    sigma    -- standard deviation of the white noise
    """

    rho = N.asarray(rho)
    p = rho.shape[0] 
    invK = N.identity(n)
    for i in range(p):
        invK -= N.diag((rho[i] / sigma) * N.ones(n-i-1), k=-i-1)
    K = inv(invK)
    Q = N.dot(K, K.T)
    if cor:
        sd = N.sqrt(N.diag(Q))
        sdout = N.multiply.outer(sd, sd)
        Q /= sd
    return Q

def ARcomponents(rho, n, drho=0.05, cor=False, sigma=1):
    """
    Numerically differentiate covariance matrices
    of AR(p) of length n with respect to AR parameters
    around the value rho.

    If drho is a vector, they are treated as steps in the numerical
    differentiation.
    """

    rho = N.asarray(rho)
    drho = N.asarray(drho)

    p = rho.shape[0]
    value = []

    if drho.shape == ():
        drho = N.ones(p, N.float) * drho
    drho = N.diag(drho)

    Q = ARcovariance(rho, n, cor=cor, sigma=sigma)
    value = [Q]
    for i in range(p):
        value.append((ARcovariance(rho + drho[i], n, cor=cor) - Q) / drho[i,i])
    return N.asarray(value)


if __name__ == "__main__":
    #print N.diag(ARcovariance([0.3], 100, cor=True), k=0)
    print len(ARcomponents([0.321],8, drho=0.02))
