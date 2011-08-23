from ._registration import _L1_moments

import numpy as np
from scipy.ndimage import gaussian_filter

TINY = float(np.finfo(np.double).tiny)
SIGMA_FACTOR = 0.05


def nonzero(x):
    """
    Force strictly positive values.
    """
    return np.maximum(x, TINY)


def dist2loss(dist, margI=None, margJ=None):
    L = dist
    LT = L.T
    if margI == None:
        margI = L.sum(0)
    if margJ == None:
        margJ = L.sum(1)
    L /= nonzero(margI)
    LT /= nonzero(margJ)
    return -np.log(nonzero(L))


class SimilarityMeasure(object):

    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.J, self.I = np.indices(shape)
        for key in kwargs.keys():
            self.__setattr__(key, kwargs[key])

    def loss(self, H):
        return np.zeros(H.shape)

    def npoints(self, H):
        return H.sum()

    def overall_loss(self, H):
        return np.sum(H * self.loss(H))

    def averaged_loss(self, H):
        return np.sum(H * self.loss(H)) / nonzero(self.npoints(H))

    def __call__(self, H):
        return -self.averaged_loss(H)


class SupervisedLikelihoodRatio(SimilarityMeasure):

    def loss(self, H):
        if not hasattr(self, 'L'):
            self.L = dist2loss(self.dist)
        return self.L


class MutualInformation(SimilarityMeasure):

    def loss(self, H):
        return dist2loss(H / nonzero(self.npoints(H)))


class ParzenMutualInformation(SimilarityMeasure):

    def loss(self, H):
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_FACTOR * np.array(H.shape)
        npts = nonzero(self.npoints(H))
        Hs = H / npts
        gaussian_filter(Hs, sigma=self.sigma, mode='constant', output=Hs)
        return dist2loss(Hs)


class DiscreteParzenMutualInformation(SimilarityMeasure):

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
    def loss(self, H):
        L = H / nonzero(self.npoints(H))
        lI = L.sum(0)
        lJ = L.sum(1)
        self.hI = lI
        self.hJ = lJ
        return -np.log(nonzero(L))

    def __call__(self, H):
        HIJ = self.averaged_loss(H)
        HI = -np.sum(self.hI * np.log(nonzero(self.hI)))
        HJ = -np.sum(self.hJ * np.log(nonzero(self.hJ)))
        return 2 * (1 - HIJ / nonzero(HI + HJ))


class CorrelationCoefficient(SimilarityMeasure):

    def loss(self, H):
        rho2 = self(H)
        I = (self.I - self.mI) / np.sqrt(nonzero(self.vI))
        J = (self.J - self.mJ) / np.sqrt(nonzero(self.vJ))
        L = rho2 * I ** 2 + rho2 * J ** 2 - 2 * self.rho * I * J
        tmp = nonzero(1. - rho2)
        L *= .5 / tmp
        L += .5 * np.log(tmp)
        return L

    def averaged_loss(self, H):
        return .5 * np.log(nonzero(1 - self(H)))

    def __call__(self, H):
        npts = nonzero(self.npoints(H))
        self.mI = np.sum(H * self.I) / npts
        self.mJ = np.sum(H * self.J) / npts
        self.vI = np.sum(H * (self.I) ** 2) / npts - self.mI ** 2
        self.vJ = np.sum(H * (self.J) ** 2) / npts - self.mJ ** 2
        self.cIJ = np.sum(H * self.J * self.I) / npts - self.mI * self.mJ
        self.rho = self.cIJ / nonzero(np.sqrt(self.vI * self.vJ))
        return self.rho ** 2


class CorrelationRatio(SimilarityMeasure):

    def loss(self, H):
        print('Sorry, not implemented yet...')
        return

    def averaged_loss(self, H):
        return .5 * np.log(nonzero(1. - self(H)))

    def __call__(self, H):
        self.npts_J = np.sum(H, 1)
        tmp = nonzero(self.npts_J)
        self.mI_J = np.sum(H * self.I, 1) / tmp
        self.vI_J = np.sum(H * (self.I) ** 2, 1) / tmp - self.mI_J ** 2
        self.npts = np.sum(self.npts_J)
        tmp = nonzero(self.npts)
        hI = np.sum(H, 0)
        hJ = np.sum(H, 1)
        self.mI = np.sum(hI * self.I[0, :]) / tmp
        self.vI = np.sum(hI * self.I[0, :] ** 2) / tmp - self.mI ** 2
        mean_vI_J = np.sum(hJ * self.vI_J) / tmp
        return 1. - mean_vI_J / nonzero(self.vI)


class CorrelationRatioL1(SimilarityMeasure):     

    def loss(self, H):
        print('Sorry, not implemented yet...')
        return

    def averaged_loss(self, H):
        return np.log(nonzero(1. - self(H)))

    def __call__(self, H):
        tmp = np.array([_L1_moments(H[j, :]) for j in range(H.shape[0])])
        self.npts_J, self.mI_J, self.sI_J = tmp[:, 0], tmp[:, 1], tmp[:, 2]
        hI = np.sum(H, 0)
        hJ = np.sum(H, 1)
        self.npts, self.mI, self.sI = _L1_moments(hI)
        mean_sI_J = np.sum(hJ * self.sI_J) / nonzero(self.npts)
        return 1. - mean_sI_J / nonzero(self.sI)


similarity_measures = {
    'slr': SupervisedLikelihoodRatio,
    'mi': MutualInformation,
    'nmi': NormalizedMutualInformation,
    'pmi': ParzenMutualInformation,
    'dpmi': DiscreteParzenMutualInformation,
    'cc': CorrelationCoefficient,
    'cr': CorrelationRatio,
    'crl1': CorrelationRatioL1}
