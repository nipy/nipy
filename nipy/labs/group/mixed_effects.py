"""
New generic implementation of multiple regression analysis under noisy
measurements. 
"""

import numpy as np 

_NITER = 2 

TINY = float(np.finfo(np.double).tiny)

def nonzero(x):
    """
    Force strictly positive values. 
    """
    return np.maximum(x, TINY)


def em(Y, VY, X, C=None, niter=_NITER, log_likelihood=False):
    """
    Maximum likelihood regression in a mixed-effect linear model using
    the EM algorithm.

    Parameters
    ----------
    Y : array
        Array of observations. 
    VY : array 
    
    C is the contrast matrix. Conventionally, C is p x q where p
    is the number of regressors. 
    
    OUTPUT: beta, s2
    beta -- array of parameter estimates
    s2 -- array of squared scale parameters.
    
    REFERENCE:
    Keller and Roche, ISBI 2008.
    """
    # Number of observations, regressors and points
    nobs = X.shape[0]
    nreg = X.shape[1]
    npts = np.prod(Y.shape[1:])
    
    # Reshape input array
    y = np.reshape(Y, (nobs, npts))
    vy = np.reshape(VY, (nobs, npts))

    # Compute the projected pseudo-inverse matrix
    if C == None:
        PpX =  np.linalg.pinv(X)
    else:
        A = np.linalg.inv(np.dot(X.T, X)) # (nreg, nreg)
        B = np.linalg.inv(np.dot(np.dot(C.T, A), C)) # (q,q)
        P = np.eye(nreg) - np.dot(np.dot(np.dot(A, C), B), C.T) # (nreg, nreg)
        PpX = np.dot(np.dot(P, A), X.T) # (nreg, nobs)

    # Initialize outputs
    b = np.zeros((nreg, npts))
    yfit = np.zeros((nobs, npts))
    s2 = np.inf

    # EM loop              
    it = 0
    while it < niter: 

        # E-step: posterior mean and variance of each "true" effect
        w1 = 1/nonzero(vy)
        w2 = 1/nonzero(s2)
        vz = 1/(w1+w2)
        z = vz*(w1*y + w2*yfit)

        # M-step: update effect and variance
        b = np.dot(PpX, z) 
        yfit = np.dot(X, b)
        s2 = np.sum((z-yfit)**2 + vz, 0)/float(nobs) 

        # Increase iteration number 
        it += 1

    # Ouput arrays
    B = np.reshape(b, [nreg] + list(Y.shape[1:]))
    S2 = np.reshape(s2, list(Y.shape[1:]))

    # Log-likelihood computation
    if not log_likelihood: 
        return B, S2
    else: 
        return B, S2, _log_likelihood(y, vy, X, b, s2)


def _log_likelihood(y, vy, X, b, s2): 
    res = y - np.dot(X, b) 
    w = nonzero(vy+s2)
    L = np.sum(np.log(w)+ res**2/w) 
    L *= -0.5
    return L



def log_likelihood_ratio(Y, VY, X, C, niter=_NITER):
    """
    Log-likelihood ratio statistic: 2*(log L - log L0) 

    It is asymptotically distributed like a chi-square with rank(C)
    degrees of freedom under the null hypothesis H0: Cb = 0. 
    """

    # Constrained log-likelihood
    B, S2, ll0 = em(Y, VY, X, C, niter, log_likelihood=True)

    # Unconstrained log-likelihood
    B, S2, ll = em(Y, VY, X, None, niter, log_likelihood=True)
    
    # -2 log R = 2*(ll-ll0)
    return np.maximum(2*(ll-ll0), 0.0)
