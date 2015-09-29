# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from numpy.linalg import inv

def ARcovariance(rho, n, cor=False, sigma=1.):
    """
    Return covariance matrix of a sample of length n from an AR(p)
    process with parameters rho.

    INPUTS:

    rho      -- an array of length p
    sigma    -- standard deviation of the white noise
    """

    rho = np.asarray(rho)
    p = rho.shape[0] 
    invK = np.identity(n)
    for i in range(p):
        invK -= np.diag((rho[i] / sigma) * np.ones(n-i-1), k=-i-1)
    K = inv(invK)
    Q = np.dot(K, K.T)
    if cor:
        sd = np.sqrt(np.diag(Q))
        sdout = np.multiply.outer(sd, sd)
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

    rho = np.asarray(rho)
    drho = np.asarray(drho)

    p = rho.shape[0]
    value = []

    if drho.shape == ():
        drho = np.ones(p, np.float) * drho
    drho = np.diag(drho)

    Q = ARcovariance(rho, n, cor=cor, sigma=sigma)
    value = [Q]
    for i in range(p):
        value.append((ARcovariance(rho + drho[i], n, cor=cor) - Q) / drho[i,i])
    return np.asarray(value)


if __name__ == "__main__":
    #print np.diag(ARcovariance([0.3], 100, cor=True), k=0)
    print(len(ARcomponents([0.321],8, drho=0.02)))
