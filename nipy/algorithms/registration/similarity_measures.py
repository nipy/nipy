from __future__ import absolute_import
from ._registration import _L1_moments

import numpy as np
from scipy.ndimage import gaussian_filter

TINY = float(np.finfo(np.double).tiny)
SIGMA_FACTOR = 0.05

# A lambda function to force positive values
nonzero = lambda x: np.maximum(x, TINY)


def correlation2loglikelihood(rho2, npts):
    """
    Re-normalize correlation.

    Convert a squared normalized correlation to a proper
    log-likelihood associated with a registration problem. The result
    is a function of both the input correlation and the number of
    points in the image overlap.

    See: Roche, medical image registration through statistical
    inference, 2001.

    Parameters
    ----------
    rho2: float
      Squared correlation measure

    npts: int
      Number of points involved in computing `rho2`

    Returns
    -------
    ll: float
      Log-likelihood re-normalized `rho2`
    """
    return -.5 * npts * np.log(nonzero(1 - rho2))


def dist2loss(q, qI=None, qJ=None):
    """
    Convert a joint distribution model q(i,j) into a pointwise loss:

    L(i,j) = - log q(i,j)/(q(i)q(j))

    where q(i) = sum_j q(i,j) and q(j) = sum_i q(i,j)

    See: Roche, medical image registration through statistical
    inference, 2001.
    """
    qT = q.T
    if qI is None:
        qI = q.sum(0)
    if qJ is None:
        qJ = q.sum(1)
    q /= nonzero(qI)
    qT /= nonzero(qJ)
    return -np.log(nonzero(q))


class SimilarityMeasure(object):
    """
    Template class
    """
    def __init__(self, shape, renormalize=False, dist=None):
        self.shape = shape
        self.J, self.I = np.indices(shape)
        self.renormalize = renormalize
        if dist is None:
            self.dist = None
        else:
            self.dist = dist.copy()

    def loss(self, H):
        return np.zeros(H.shape)

    def npoints(self, H):
        return H.sum()

    def __call__(self, H):
        total_loss = np.sum(H * self.loss(H))
        if not self.renormalize:
            total_loss /= nonzero(self.npoints(H))
        return -total_loss


class SupervisedLikelihoodRatio(SimilarityMeasure):
    """
    Assume a joint intensity distribution model is given by self.dist
    """
    def loss(self, H):
        if not hasattr(self, 'L'):
            if self.dist is None:
                raise ValueError('SupervisedLikelihoodRatio: dist attribute cannot be None')
            if not self.dist.shape == H.shape:
                raise ValueError('SupervisedLikelihoodRatio: wrong shape for dist attribute')
            self.L = dist2loss(self.dist)
        return self.L


class MutualInformation(SimilarityMeasure):
    """
    Use the normalized joint histogram as a distribution model
    """
    def loss(self, H):
        return dist2loss(H / nonzero(self.npoints(H)))


class ParzenMutualInformation(SimilarityMeasure):
    """
    Use Parzen windowing to estimate the distribution model
    """
    def loss(self, H):
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_FACTOR * np.array(H.shape)
        npts = nonzero(self.npoints(H))
        Hs = H / npts
        gaussian_filter(Hs, sigma=self.sigma, mode='constant', output=Hs)
        return dist2loss(Hs)


class DiscreteParzenMutualInformation(SimilarityMeasure):
    """
    Use Parzen windowing in the discrete case to estimate the
    distribution model
    """
    def loss(self, H):
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_FACTOR * np.array(H.shape)
        Hs = gaussian_filter(H, sigma=self.sigma, mode='constant')
        Hs /= nonzero(Hs.sum())
        return dist2loss(Hs)


class NormalizedMutualInformation(SimilarityMeasure):
    """
    NMI = 2*(1 - H(I,J)/[H(I)+H(J)])
        = 2*MI/[H(I)+H(J)])
    """
    def __call__(self, H):
        H = H / nonzero(self.npoints(H))
        hI = H.sum(0)
        hJ = H.sum(1)
        entIJ = -np.sum(H * np.log(nonzero(H)))
        entI = -np.sum(hI * np.log(nonzero(hI)))
        entJ = -np.sum(hJ * np.log(nonzero(hJ)))
        return 2 * (1 - entIJ / nonzero(entI + entJ))


class CorrelationCoefficient(SimilarityMeasure):
    """
    Use a bivariate Gaussian as a distribution model
    """
    def loss(self, H):
        rho2 = self(H)
        I = (self.I - self.mI) / np.sqrt(nonzero(self.vI))
        J = (self.J - self.mJ) / np.sqrt(nonzero(self.vJ))
        L = rho2 * I ** 2 + rho2 * J ** 2 - 2 * self.rho * I * J
        tmp = nonzero(1. - rho2)
        L *= .5 / tmp
        L += .5 * np.log(tmp)
        return L

    def __call__(self, H):
        npts = nonzero(self.npoints(H))
        mI = np.sum(H * self.I) / npts
        mJ = np.sum(H * self.J) / npts
        vI = np.sum(H * (self.I) ** 2) / npts - mI ** 2
        vJ = np.sum(H * (self.J) ** 2) / npts - mJ ** 2
        cIJ = np.sum(H * self.J * self.I) / npts - mI * mJ
        rho2 = (cIJ / nonzero(np.sqrt(vI * vJ))) ** 2
        if self.renormalize:
            rho2 = correlation2loglikelihood(rho2, npts)
        return rho2


class CorrelationRatio(SimilarityMeasure):
    """
    Use a nonlinear regression model with Gaussian errors as a
    distribution model
    """
    def __call__(self, H):
        npts_J = np.sum(H, 1)
        tmp = nonzero(npts_J)
        mI_J = np.sum(H * self.I, 1) / tmp
        vI_J = np.sum(H * (self.I) ** 2, 1) / tmp - mI_J ** 2
        npts = np.sum(npts_J)
        tmp = nonzero(npts)
        hI = np.sum(H, 0)
        hJ = np.sum(H, 1)
        mI = np.sum(hI * self.I[0, :]) / tmp
        vI = np.sum(hI * self.I[0, :] ** 2) / tmp - mI ** 2
        mean_vI_J = np.sum(hJ * vI_J) / tmp
        eta2 = 1. - mean_vI_J / nonzero(vI)
        if self.renormalize:
            eta2 = correlation2loglikelihood(eta2, npts)
        return eta2


class CorrelationRatioL1(SimilarityMeasure):
    """
    Use a nonlinear regression model with Laplace distributed errors
    as a distribution model
    """
    def __call__(self, H):
        moments = np.array([_L1_moments(H[j, :]) for j in range(H.shape[0])])
        npts_J, mI_J, sI_J = moments[:, 0], moments[:, 1], moments[:, 2]
        hI = np.sum(H, 0)
        hJ = np.sum(H, 1)
        npts, mI, sI = _L1_moments(hI)
        mean_sI_J = np.sum(hJ * sI_J) / nonzero(npts)
        eta2 = 1. - mean_sI_J / nonzero(sI)
        if self.renormalize:
            eta2 = correlation2loglikelihood(eta2, npts)
        return eta2


similarity_measures = {
    'slr': SupervisedLikelihoodRatio,
    'mi': MutualInformation,
    'nmi': NormalizedMutualInformation,
    'pmi': ParzenMutualInformation,
    'dpmi': DiscreteParzenMutualInformation,
    'cc': CorrelationCoefficient,
    'cr': CorrelationRatio,
    'crl1': CorrelationRatioL1}
