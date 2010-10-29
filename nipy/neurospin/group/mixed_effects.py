"""
New generic implementation of multiple regression analysis under noisy
measurements. 
"""

_NITER = 2 

import numpy as np 

def def em(ndarray Y, ndarray VY, ndarray X, ndarray C=None, niter=_NITER):
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

    # Precompute
    




