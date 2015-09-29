"""
Generic implementation of multiple regression analysis under noisy
measurements.
"""
from __future__ import absolute_import

import numpy as np

nonzero = lambda x: np.maximum(x, 1e-25)


def two_level_glm(y, vy, X, niter=10):
    """
    Inference of a mixed-effect linear model using the variational
    Bayes algorithm.

    Parameters
    ----------
    y : array-like
      Array of observations. Shape should be (n, ...) where n is the
      number of independent observations per unit.

    vy : array-like
      First-level variances associated with the observations. Should
      be of the same shape as Y.

    X : array-like
      Second-level design matrix. Shape should be (n, p) where n is
      the number of observations per unit, and p is the number of
      regressors.

    Returns
    -------
    beta : array-like
      Effect estimates (posterior means)

    s2 : array-like
      Variance estimates. The posterior variance matrix of beta[:, i]
      may be computed by s2[:, i] * inv(X.T * X)

    dof : float
      Degrees of freedom as per the variational Bayes approximation
      (simply, the number of observations minus the number of
      independent regressors)
    """
    # Number of observations, regressors and points
    nobs = X.shape[0]
    if X.ndim == 1:
        nreg = 1
    else:
        nreg = X.shape[1]
    if nobs <= nreg:
        raise ValueError('Too many regressors compared to data size')
    if y.ndim == 1:
        npts = 1
    else:
        npts = np.prod(y.shape[1:])

    # Reshape input arrays
    X = X.reshape((nobs, nreg))
    y = np.reshape(y, (nobs, npts))
    vy = nonzero(np.reshape(vy, (nobs, npts)))

    # Degrees of freedom
    dof = float(nobs - nreg)

    # Compute the pseudo-inverse matrix
    pinvX = np.linalg.pinv(X)

    # Initialize outputs
    b = np.zeros((nreg, npts))
    zfit = np.zeros((nobs, npts))
    s2 = np.inf

    # VB loop
    for it in range(niter):

        # Update distribution of "true" effects
        w1 = 1 / vy
        w2 = 1 / nonzero(s2)
        vz = 1 / (w1 + w2)
        z = vz * (w1 * y + w2 * zfit)

        # Update distribution of population parameters
        b = np.dot(pinvX, z)
        zfit = np.dot(X, b)
        s2 = np.sum((z - zfit) ** 2 + vz, 0) / dof

    # Ouput arrays
    B = np.reshape(b, [nreg] + list(y.shape[1:]))
    S2 = np.reshape(s2, list(y.shape[1:]))

    return B, S2, dof
