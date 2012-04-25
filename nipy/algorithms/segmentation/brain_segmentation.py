import numpy as np

from .segmentation import (Segmentation,
                           moment_matching,
                           map_from_ppm)

T1_ref_class_parameters = {}
T1_ref_class_parameters['3k'] = {
    'mu': np.array([813.9, 1628.4, 2155.8]),
    'sigma': np.array([46483.36, 30241.21, 17134.81]),
    'glob_mu': 1643.1,
    'glob_sigma': 252807.84}
T1_ref_class_parameters['4k'] = {
    'mu': np.array([112.287586, 230.84312703,
                    305.91563692, 379.11225839]),
    'sigma': np.array([2265.97060371, 1222.01054703,
                        822.11289473, 611.15921583]),
    'glob_mu': 271.88542994875456,
    'glob_sigma': 9150.47573161447}
T1_ref_class_parameters['4kpv'] = T1_ref_class_parameters['4k']


class BrainT1Segmentation(object):

    def __init__(self, data, mask=None, model='3k'):

        if not model in T1_ref_class_parameters.keys():
            raise ValueError('unknown brain segmentation model')

        self.labels = ('CSF', 'GM', 'WM')
        self.data = data
        self.mask = mask

        self.mu = T1_ref_class_parameters[model]['mu']
        self.sigma = T1_ref_class_parameters[model]['sigma']
        self.glob_mu = T1_ref_class_parameters[model]['glob_mu']
        self.glob_sigma = T1_ref_class_parameters[model]['glob_sigma']

        if model == '3k':
            self.mixmat = np.eye(3)
        elif model == '4k':
            self.mixmat = np.array([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]])
        elif model == '4kpv':
            self.mixmat = np.array([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., .5, .5],
                                    [0., 0., 1.]])
        else:
            raise ValueError('unknown brain segmentation model')

    def init_parameters(self):
        if not self.mask == None:
            data = self.data[self.mask]
        else:
            data = self.data
        return moment_matching(data,
                               self.mu, self.sigma,
                               self.glob_mu, self.glob_sigma)

    def run(self, niters=25, beta=0.5):
        mu, sigma = self.init_parameters()
        S = Segmentation(self.data, mask=self.mask,
                         mu=mu, sigma=sigma,
                         beta=beta)
        S.run(niters=niters)
        self.ppm = np.dot(S.ppm, self.mixmat)
        self.label = map_from_ppm(self.ppm, self.mask)
