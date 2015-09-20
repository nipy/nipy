from __future__ import absolute_import
import numpy as np

from .segmentation import (Segmentation,
                           moment_matching,
                           map_from_ppm)

T1_ref_params = {}
T1_ref_params['glob_mu'] = 1643.2
T1_ref_params['glob_sigma'] = 252772.3
T1_ref_params['3k'] = {
    'mu': np.array([813.9, 1628.3, 2155.8]),
    'sigma': np.array([46499.0, 30233.4, 17130.0])}
T1_ref_params['4k'] = {
    'mu': np.array([816.1, 1613.7, 1912.3, 2169.3]),
    'sigma': np.array([47117.6, 27053.8, 8302.2, 14970.8])}
T1_ref_params['5k'] = {
    'mu': np.array([724.2, 1169.3, 1631.5, 1917.0, 2169.2]),
    'sigma': np.array([22554.8, 21368.9, 20560.1, 7302.6, 14962.1])}


class BrainT1Segmentation(object):

    def __init__(self, data, mask=None, model='3k',
                 niters=25, ngb_size=6, beta=0.5,
                 ref_params=None, init_params=None,
                 convert=True):

        self.labels = ('CSF', 'GM', 'WM')
        self.data = data
        self.mask = mask

        mixmat = np.asarray(model)
        if mixmat.ndim == 2:
            nclasses = mixmat.shape[0]
            if nclasses < 3:
                raise ValueError('at least 3 classes required')
            if not mixmat.shape[1] == 3:
                raise ValueError('mixing matrix should have 3 rows')
            self.mixmat = mixmat
        elif model == '3k':
            self.mixmat = np.eye(3)
        elif model == '4k':
            self.mixmat = np.array([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]])
        elif model == '5k':
            self.mixmat = np.array([[1., 0., 0.],
                                    [1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]])
        else:
            raise ValueError('unknown brain segmentation model')

        self.niters = int(niters)
        self.beta = float(beta)
        self.ngb_size = int(ngb_size)

        # Class parameter initialization
        if init_params is None:
            if ref_params is None:
                ref_params = T1_ref_params
            self.init_mu, self.init_sigma = self._init_parameters(ref_params)
        else:
            self.init_mu = np.array(init_params[0], dtype='double')
            self.init_sigma = np.array(init_params[1], dtype='double')
            if not len(self.init_mu) == self.mixmat.shape[0]\
                    or not len(self.init_sigma) == self.mixmat.shape[0]:
                raise ValueError('Inconsistent initial parameter estimates')

        self._run()
        if convert:
            self.convert()
        else:
            self.label = map_from_ppm(self.ppm, self.mask)

    def _init_parameters(self, ref_params):

        if self.mask is not None:
            data = self.data[self.mask]
        else:
            data = self.data

        nclasses = self.mixmat.shape[0]
        if nclasses <= 5:
            key = str(self.mixmat.shape[0]) + 'k'
            ref_mu = ref_params[key]['mu']
            ref_sigma = ref_params[key]['sigma']
        else:
            ref_mu = np.linspace(ref_params['3k']['mu'][0],
                                 ref_params['3k']['mu'][-1],
                                 num=nclasses)
            ref_sigma = np.linspace(ref_params['3k']['sigma'][0],
                                    ref_params['3k']['sigma'][-1],
                                    num=nclasses)

        return moment_matching(data, ref_mu, ref_sigma,
                               ref_params['glob_mu'],
                               ref_params['glob_sigma'])

    def _run(self):
        S = Segmentation(self.data, mask=self.mask,
                         mu=self.init_mu, sigma=self.init_sigma,
                         ngb_size=self.ngb_size, beta=self.beta)
        S.run(niters=self.niters)
        self.mu = S.mu
        self.sigma = S.sigma
        self.ppm = S.ppm

    def convert(self):
        if self.ppm.shape[-1] == self.mixmat.shape[0]:
            self.ppm = np.dot(self.ppm, self.mixmat)
            self.label = map_from_ppm(self.ppm, self.mask)
