# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from ._segmentation import _ve_step, _interaction_energy


TINY = 1e-300
NITERS = 20
BETA = 0.2
VERBOSE = True


def print_(s):
    if VERBOSE:
        print(s)


def gauss_dist(x, mu, sigma):
    return np.exp(-.5 * ((x - float(mu)) / float(sigma)) ** 2) / float(sigma)


def laplace_dist(x, mu, sigma):
    return np.exp(-np.abs((x - float(mu)) / float(sigma))) / float(sigma)


def vm_step_gauss(ppm, data_masked, mask):
    """
    ppm: ndarray (4d)
    data_masked: ndarray (1d, masked data)
    mask: 3-element tuple of 1d ndarrays (X,Y,Z)
    """
    nclasses = ppm.shape[-1]
    mu = np.zeros(nclasses)
    sigma = np.zeros(nclasses)
    prop = np.zeros(nclasses)
    for i in range(nclasses):
        P = ppm[..., i][mask]
        Z = P.sum()
        tmp = data_masked * P
        mu[i] = tmp.sum() / Z
        sigma[i] = np.sqrt(np.sum(tmp * data_masked) / Z - mu[i] ** 2)
        prop[i] = Z / float(data_masked.size)
    return mu, sigma, prop


def weighted_median(x, w, ind):
    F = np.cumsum(w[ind])
    f = .5 * (w.sum() + 1)
    i = np.searchsorted(F, f)
    if i == 0:
        return x[ind[0]]
    wr = (f - F[i - 1]) / (F[i] - F[i - 1])
    jr = ind[i]
    jl = ind[i - 1]
    return wr * x[jr] + (1 - wr) * x[jl]


def vm_step_laplace(ppm, data_masked, mask):
    """
    ppm: ndarray (4d)
    data_masked: ndarray (1d, masked data)
    mask: 3-element tuple of 1d ndarrays (X,Y,Z)
    """
    nclasses = ppm.shape[-1]
    mu = np.zeros(nclasses)
    sigma = np.zeros(nclasses)
    prop = np.zeros(nclasses)
    sorted_indices = np.argsort(data_masked)  # data_masked[ind] increasing
    for i in range(nclasses):
        P = ppm[..., i][mask]
        mu[i] = weighted_median(data_masked, P, sorted_indices)
        sigma[i] = np.sum(np.abs(P * (data_masked - mu[i]))) / P.sum()
        prop[i] = P.sum() / float(data_masked.size)
    return mu, sigma, prop


message_passing = {'mf': 0, 'icm': 1, 'bp': 2}
noise_model = {'gauss': (gauss_dist, vm_step_gauss),
               'laplace': (laplace_dist, vm_step_laplace)}


class VEM(object):
    """
    Classification via VEM algorithm.
    """
    def __init__(self, data, labels, mask=None, noise='gauss',
                 ppm=None, synchronous=False, scheme='mf'):
        """
        A class to represent a variational EM algorithm for tissue
        classification.

        Parameters
        ----------
        data: array
          Image data (n-dimensional)
        labels: int or sequence
          Desired number of classes, or sequence of strings
        mask: sequence
          Sequence of one-dimensional coordinate arrays
        """
        # Labels
        if hasattr(labels, '__iter__'):
            self.nclasses = len(labels)
            self.labels = labels
        else:
            self.nclasses = int(labels)
            self.labels = [str(l) for l in range(self.nclasses)]

        # Make default mask (required by MRF regularization) This will
        # be passed to the _ve_step C-routine, which assumes a
        # contiguous int array and raise an error otherwise.  Voxels
        # on the image borders are further rejected to avoid
        # segmentation faults.
        if mask == None:
            XYZ = np.mgrid[[slice(1, s - 1) for s in data.shape]]
            XYZ = np.reshape(XYZ, (XYZ.shape[0], np.prod(XYZ.shape[1::]))).T
            XYZ = np.asarray(XYZ, dtype='int', order='C')
        else:
            submask = (mask[0] > 0) * (mask[0] < data.shape[0] - 1)
            for i in range(1, len(mask)):
                submask *= (mask[i] > 0) * (mask[i] < data.shape[i] - 1)
            mask = [mask[i][submask] for i in range(len(mask))]
            XYZ = np.zeros((len(mask[0]), len(mask)), dtype='int')
            for i in range(len(mask)):
                XYZ[:, i] = mask[i]
        self._XYZ = XYZ
        self.mask = tuple(XYZ.T)

        # If a ppm is provided, interpret it as a prior, otherwise
        # create ppm from scratch and assume flat prior.
        if ppm == None:
            self.ppm = np.zeros(list(data.shape) + [self.nclasses])
            self.ppm[self.mask] = 1. / self.nclasses
        else:
            self.ppm = ppm
        self.data_masked = data[self.mask]
        self.prior_ext_field = self.ppm[self.mask]
        self.posterior_ext_field = np.zeros([self.data_masked.size,
                                             self.nclasses])
        # Inference scheme parameters
        self.synchronous = synchronous
        if scheme in message_passing:
            self.scheme = message_passing[scheme]
        else:
            raise ValueError('Unknown message passing scheme')
        if noise in noise_model:
            self.dist, self._vm_step = noise_model[noise]
        else:
            raise ValueError('Unknown noise model')

        # Cache beta parameter
        self._beta = BETA

    # VM-step: estimate parameters
    def vm_step(self):
        """
        Return (mu, sigma)
        """
        return self._vm_step(self.ppm, self.data_masked, self.mask)

    def sort_labels(self, mu):
        """
        Sort the array labels to match mean tissue intensities ``mu``.
        """
        K = len(mu)
        tmp = np.asarray(self.labels)
        labels = np.zeros(K, dtype=tmp.dtype)
        labels[np.argsort(mu)] = tmp
        return list(labels)

    # VE-step: update tissue probability map
    def ve_step(self, mu, sigma, prop=None, beta=BETA):
        """
        VE-step
        """
        # Cache beta parameter
        self._beta = beta

        # Compute complete-data likelihood maps, replacing very small
        # values for numerical stability
        for i in range(self.nclasses):
            self.posterior_ext_field[:, i] = self.prior_ext_field[:, i] * \
                self.dist(self.data_masked, mu[i], sigma[i])
            if not prop == None:
                self.posterior_ext_field[:, i] *= prop[i]
        self.posterior_ext_field[:] = np.maximum(self.posterior_ext_field,
                                                 TINY)

        # Normalize reference probability map
        if beta == 0.0 or self._XYZ.shape[1] != 3:
            self.ppm[self.mask] = (self.posterior_ext_field.T /
                                   self.posterior_ext_field.sum(1)).T

        # Update and normalize reference probabibility map using
        # neighborhood information (mean-field theory)
        else:
            print_('  ... MRF regularization')
            self.ppm = _ve_step(self.ppm, self.posterior_ext_field, self._XYZ,
                                beta, self.synchronous, self.scheme)

    def run(self, mu=None, sigma=None, prop=None, beta=BETA,
            niters=NITERS, freeze_prop=True):

        do_vm_step = (mu == None)

        def check(x, default=0.0):
            if x == None:
                return default * np.ones(self.nclasses, dtype='double')
            else:
                return np.asarray(x, dtype='double')
        mu = check(mu)
        sigma = check(sigma)
        prop = check(prop, default=1. / self.nclasses)
        prop0 = prop

        for i in range(niters):
            print_('VEM iter %d/%d' % (i + 1, niters))
            print_('  VM-step...')
            if do_vm_step:
                mu, sigma, prop = self.vm_step()
                if freeze_prop:  # account for label switching
                    prop = prop0[np.argsort(mu)]
            print_('  VE-step...')
            self.ve_step(mu, sigma, prop, beta=beta)
            do_vm_step = True

        return mu, sigma, prop

    def free_energy(self):
        """
        Compute the free energy defined as:

        F(q, theta) = int q(x) log q(x)/p(x,y/theta) dx

        associated with input parameters mu,
        sigma and beta (up to an ignored constant).
        """
        q = self.ppm[self.mask]
        # Entropy term
        f = np.sum(q * np.log(np.maximum(q / self.posterior_ext_field, TINY)))
        # Interaction term
        if self._beta > 0.0:
            fc = _interaction_energy(self.ppm, self._XYZ)
            f -= .5 * self._beta * fc
        return f
