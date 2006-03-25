"""
This module defines a class to output estimates
of delays and contrasts of delays.

Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H., Evans, A.C. (2002). \'Estimating the delay of the response in fMRI data.\' NeuroImage, 16:593-606.

"""

import copy, os, csv, string, fpformat
import numpy as N
import enthought.traits as traits
import neuroimaging.image as image
from neuroimaging.reference import grid
from neuroimaging.fmri.regression import TContrastOutput 
from neuroimaging.fmri import FContrastOutput
from neuroimaging.statistics import utils
from neuroimaging.fmri.protocol import ExperimentalQuantitative

class DelayContrast(FContrastOutput):

    IRF = traits.Any()
    dt = traits.Float(0.01)
    delta = traits.Array(N.linspace(-4.5,4.5,91))
    Tmax = 100.
    Tmin = 100.

    """
    Output a delay contrast.

    Delay contrasts are specified by a contrast whose (design space) columns are expected to
    NOT be convolved with any HRF. They will be convolved with self.IRF which is expected
    to be a filter with a canonical HRF and its derivative -- defaults to the Glover model.

    If the contrast is real-valued, representing
    one column in a Formula object, then the delay for 
    this particular column is output.

    If the contrast is multi-valued, then a
    contrast of delays is computed. The contrast is formed
    by assigning a weight of 1. to each column in the function of time.


    TO DO: check that the convolved term is actually in the design column space.

    """

    def __init__(self, fmri_image, contrast, path='.',
                 IRF=hrf.HRF(deriv=True), **keywords):
        FContrastOutput.__init__(self, fmri_image, contrast, path=path, **keywords)
        self.delayapprox = self.IRF.deltaPCA(self.delta, dt=self.dt,
                                             tmax=50., lower=-15.0, spectral=True)

        self.setup_contrast()
        
    def setup_contrast(self):
        """
        Setup the contrast to output its convolution with self.IRF -- writes
        over the previous contrast's term attribute.
        """

        t = N.arange(0, max(self.fmri_image.frametimes), self.dt)
        v = N.asarray(self.contrast.term(time=t))
        fns = []

        for i in range(v.shape[0]): # check this order here!
            fn = scipy.interpolate.interp1d(t, v[i])
            fns.append(fn)

        self.contrast.term = ExperimentalQuantitative(self.contrast.term.termname, fns)
        self.contrast.term.convolve(self.IRF)
        self.contrast.getmatrix(time=self.fmri_image.frametimes)

        self.effectmatrix = self.contrast.matrix[0::2]
        self.deltamatrix = self.contrast.matrix[1::2]

    def effect(self, results):

        delay = self.delayapprox

        self.gamma0 = N.dot(self.effectmatrix, results.beta)
        self.gamma1 = N.dot(self.deltamatrix, results.beta)

        nrow = self.gamma0.shape[0]
        self.T0sq = N.zeros(gamma0.shape, N.Float)
        
        for i in range(nrow):
            sefl.T0sq[i] = (self.gamma0[i]**2 *
                            utils.inv(results.cov_beta(matrix=self.effectmatrix[i])))

        self.r = self.gamma1 * utils.inv2(self.gamma0)
        self.rC = self.r * self.T0sq / (1. + self.T0sq)
        self.deltahat = delay.inverse(self.rC)
        self.effect = N.add.reduce(deltahat, axis=0)

    def sd(self, results):

        delay = self.delayapprox

        self.T1 = N.zeros(gamma0.shape, N.Float)

        for i in range(nrow):
            self.T1[i] = self.gamma1[i] * utils.inv(N.sqrt(results.cov_beta(matrix=self.deltamatrix[i])))

        a1 = 1 + 1. * utils.inv(self.T0sq)

        gdot = N.array(([self.r * (a1 - 2.) * utils.inv2(self.gamma0 * a1**2),
                              utils.inv2(self.gamma0 * a1)] *
                             utils.inv2(delay.dforward(self.deltahat))))

        tmpcov = N.zeros((2*nrow,)*2 + self.TOsq.shape[1:], N.Float)

        nrow = self.effectmatrix.shape[0]
        for i in range(nrow):
            for j in range(i+1):
                tmpcov[i,j] = results.cov_beta(matrix=self.effectmatrix[i],
                                                other=self.effectmatrix[j])
                tmpcov[nrow+i,j] = results.cov_beta(matrix=self.deltamatrix[i],
                                                     other=self.effectmatrix[j])
                tmpcov[nrow+i,nrow+j] = results.cov_beta(matrix=self.deltamatrix[i],
                                                          other=self.deltamatrix[j])
            tmpcov[j,i] = tmpcov[i,j]
            tmpcov[j,nrow+i] = tmpcov[nrow+i,j]
            tmpcov[nrow+i,nrow+j] = tmpcov[nrow+j,nrow+i]
            
        cov = N.zeros((nrow,)*2 + self.T0sq.shape[1:], N.Float)

        for i in range(nevents):
            gdoti = gdot[:,i]
            for j in range(i + 1):
                gdotj = gdot[:,j]
                cov[i,j] = (gdoti[0] * gdotj[0] * results.cov_beta(matrix=self.effectmatrix[i],
                                                                   other=self.effectmatrix[j]) +  
                            gdoti[0] * gdotj[1] * results.cov_beta(matrix=self.effectmatrix[i],
                                                                   other=self.deltamatrix[j]) +
                            gdoti[1] * gdotj[0] * results.cov_beta(matrix=self.deltamatrix[i],
                                                                   other=self.effectmatrix[j]) +
                            gdoti[1] * gdotj[1] * results.cov_beta(matrix=self.deltamatrix[i],
                                                                   other=self.deltamatrix[j]))
            cov[j,i] = cov[i,j]

            var = N.add.reduce(N.add.reduce(cov, axis=0), axis=0)
            self.sd = N.sqrt(var)                

    def t(results):
        self.t = self.effect * utils.inv(self.sd)
        self.t = clip(self.t, self.Tmin, self.Tmax)

