import numpy as np

from .segmentation import (Segmentation,
                           moment_matching,
                           map_from_ppm)

T1_ref_params = {}
T1_ref_params['3k'] = {
    'mu': np.array([813.9, 1628.4, 2155.8]),
    'sigma': np.array([46483.4, 30241.2, 17134.8]),
    'glob_mu': 1643.1,
    'glob_sigma': 252807.8}
T1_ref_params['4k'] = {
    'mu': np.array([788.5, 1544.6, 1734.8, 2159.7]),
    'sigma': np.array([38366.7, 30788.6, 19312.4, 16399.7])}
T1_ref_params['5k'] = {
    'mu': np.array([722.8, 1152.6, 1556.4, 1728.8, 2165.0]),
    'sigma': np.array([22570.7, 22844.6, 18808.2, 20793.6, 15554.7])}
T1_ref_params['glob_mu'] = 1643.2
T1_ref_params['glob_sigma'] = 252772.3


class BrainT1Segmentation(object):

    def __init__(self, data, mask=None, model='3k'):

        self.labels = ('CSF', 'GM', 'WM')
        self.data = data
        self.mask = mask

        mixmat = np.asarray(model)
        if mixmat.ndim == 2:
            nclasses = mixmat.shape[0]
            if nclasses < 3:
                raise ValueError('brain segmentation requires at least 3 classes')
            if not mixmat.shape[1] == 3:
                raise ValueError('mixing matrix should have 3 rows')
            self.mixmat = mixmat
        elif model in T1_ref_params.keys():
            if model == '3k':
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
            nclasses = self.mixmat.shape[0]
        else:
            raise ValueError('unknown brain segmentation model')

        if nclasses <= 5:
            key = str(self.mixmat.shape[0]) + 'k'
            self.ref_mu = T1_ref_params[key]['mu']
            self.ref_sigma = T1_ref_params[key]['sigma']
        else:
            self.ref_mu = np.linspace(T1_ref_params['3k']['mu'][0],
                                      T1_ref_params['3k']['mu'][-1],
                                      num=nclasses)
            self.ref_sigma = np.linspace(T1_ref_params['3k']['sigma'][0],
                                         T1_ref_params['3k']['sigma'][-1],
                                         num=nclasses)
        self.glob_mu = T1_ref_params['glob_mu']
        self.glob_sigma = T1_ref_params['glob_sigma']

    def init_parameters(self):
        if not self.mask == None:
            data = self.data[self.mask]
        else:
            data = self.data
        return moment_matching(data,
                               self.ref_mu, self.ref_sigma,
                               self.glob_mu, self.glob_sigma)

    def run(self, niters=25, ngb_size=6, beta=0.5, convert=True):
        self.init_mu, self.init_sigma = self.init_parameters()
        S = Segmentation(self.data, mask=self.mask,
                         mu=self.init_mu, sigma=self.init_sigma,
                         ngb_size=ngb_size, beta=beta)
        S.run(niters=niters)
        self.mu = S.mu
        self.sigma = S.sigma
        if convert:
            self.ppm = np.dot(S.ppm, self.mixmat)
        else:
            self.ppm = S.ppm
        self.label = map_from_ppm(self.ppm, self.mask)
