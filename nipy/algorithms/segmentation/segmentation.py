# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import gc
import numpy as np
from ._segmentation import (_ve_step,
                            _gen_ve_step)

TINY = 1e-50
HUGE = 1e50
NITERS = 10


class Segmentation(object):

    def __init__(self, data, mu=None, sigma=None,
                 ppm=None, prior=None, U=None, beta=0,
                 bottom_corner=(0, 0, 0), top_corner=(0, 0, 0),
                 mask=None):

        data = data.squeeze()
        if not len(data.shape) in (3, 4):
            raise ValueError('Invalid input image')
        if len(data.shape) == 3:
            nchannels = 1
            space_shape = data.shape
        else:
            nchannels = data.shape[-1]
            space_shape = data.shape[0:-1]

        self.nchannels = nchannels

        # Make default mask (required by MRF regularization). This wil
        # be passed to the _ve_step C-routine, which assumes a
        # contiguous int array and raise an error otherwise. Voxels on
        # the image borders are further rejected to avoid segmentation
        # faults.
        if mask == None:
            mask = [slice(max(bc, 1), s - max(tc, 1))\
                        for s, bc, tc in zip(space_shape,
                                             bottom_corner,
                                             top_corner)]
            XYZ = np.mgrid[mask]
            XYZ = np.reshape(XYZ, (XYZ.shape[0], np.prod(XYZ.shape[1::]))).T
            self.XYZ = np.asarray(XYZ, dtype='int', order='C')
        else:
            submask = (mask[0] > 0) * (mask[0] < space_shape[0] - 1)
            for i in range(1, len(mask)):
                submask *= (mask[i] > 0) * (mask[i] < space_shape[i] - 1)
            mask = [mask[i][submask] for i in range(len(mask))]
            data_msk = data[mask]
            XYZ = np.zeros((len(mask[0]), len(mask)), dtype='int')
            for i in range(len(mask)):
                XYZ[:, i] = mask[i]
            self.XYZ = XYZ

        self.mask = mask
        data_msk = data[mask]

        if nchannels == 1:
            self.data = data_msk.reshape((np.prod(data_msk.shape), 1))
        else:
            self.data = data_msk.reshape((np.prod(data_msk.shape[0:-1]),
                                          data_msk.shape[-1]))

        # If no initial ppm is provided, assume sensible `mu` and
        # `sigma` are provided
        if ppm == None:
            nclasses = len(mu)
            self.ppm = np.zeros(list(space_shape) + [nclasses])
            self.is_ppm = False
            self.mu = np.asarray(mu, dtype='double').reshape(\
                (nclasses, nchannels))
            self.sigma = np.asarray(sigma, dtype='double').reshape(\
                (nclasses, nchannels, nchannels))
        elif mu == None:
            nclasses = ppm.shape[-1]
            self.ppm = np.asarray(ppm)
            self.is_ppm = True
            self.mu = np.zeros((nclasses, nchannels))
            self.sigma = np.zeros((nclasses, nchannels, nchannels))
        else:
            raise ValueError('missing information')
        self.nclasses = nclasses
        self.ext_field = np.zeros([self.data.shape[0], nclasses])

        if not prior == None:
            self.prior = np.asarray(prior)[self.mask].reshape(\
                [self.data.shape[0], nclasses])
        else:
            self.prior = None

        self.set_energy(U, beta)

        # Should check whether input data is consistent with parameter
        # sizes

    def set_energy(self, U, beta):
        if not U == None:
            self.U = np.asarray(U).copy()  # make sure it's C-contiguous
        else:
            self.U = None
        self.beta = float(beta)

    def vm_step(self, freeze=()):

        print(' VM step...')

        classes = range(self.nclasses)
        for i in freeze:
            classes.remove(i)

        for i in classes:
            P = self.ppm[..., i][self.mask].ravel()
            Z = np.maximum(TINY, P.sum())
            tmp = self.data.T * P.T
            mu = tmp.sum(1) / Z
            mu_ = mu.reshape((len(mu), 1))
            sigma = np.dot(tmp, self.data) / Z - np.dot(mu_, mu_.T)
            self.mu[i] = mu
            self.sigma[i] = sigma

            print('*******************')
            print P.sum(-1)
            print Z, mu, sigma

        gc.enable()
        gc.collect()

    def ve_step(self):

        print(' VE step...')

        # Compute posterior external field (no voxel interactions)
        for i in range(self.nclasses):
            print('  tissue %d' % i)
            centered_data = self.data - self.mu[i]
            if self.nchannels == 1:
                inv_sigma = 1. / np.maximum(TINY, self.sigma[i])
                norm_factor = np.sqrt(inv_sigma.squeeze())
            else:
                inv_sigma = np.linalg.inv(self.sigma[i])
                norm_factor = 1. / np.sqrt(\
                    np.maximum(TINY, np.linalg.det(self.sigma[i])))
            maha = np.sum(centered_data * np.dot(inv_sigma,
                                                 centered_data.T).T, 1)
            self.ext_field[:, i] = np.exp(-.5 * maha)
            self.ext_field[:, i] *= norm_factor

        if not self.prior == None:
            self.ext_field *= self.prior
        self.ext_field.clip(TINY, HUGE, out=self.ext_field)

        if self.beta == 0:
            print('  ... Normalizing...')
            tmp = self.ext_field.T
            tmp /= tmp.sum(0)
            self.ppm[self.mask] = self.ext_field.reshape(\
                self.ppm[self.mask].shape)
        else:
            print('  ... MRF regularization')
            if self.U == None:
                self.ppm = _ve_step(self.ppm, self.ext_field, self.XYZ,
                                    self.beta, False, 0)
            else:
                self.ppm = _gen_ve_step(self.ppm, self.ext_field, self.XYZ,
                                        self.U, self.beta)

        gc.enable()
        gc.collect()

    def run(self, niters=NITERS, freeze=()):

        if self.is_ppm:
            self.vm_step(freeze=freeze)
        for i in range(niters):
            print(' Iter %d/%d...' % (i + 1, niters))
            self.ve_step()
            self.vm_step(freeze=freeze)
        self.is_ppm = True


def maximum_a_posteriori(ppm):
    return ppm.argmax(-1)
