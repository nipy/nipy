"""
This module defines a class to output estimates
of delays and contrasts of delays.

Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H., Evans, A.C. (2002). \'Estimating the delay of the response in fMRI data.\' NeuroImage, 16:593-606.

"""

import neuroimaging
import copy, os, csv, string, fpformat, types
import numpy as N
import enthought.traits as traits
import neuroimaging.image as image
from neuroimaging.reference import grid
from neuroimaging.fmri.regression import TContrastOutput 
from neuroimaging.statistics import utils, regression, contrast
from neuroimaging.fmri.protocol import ExperimentalRegressor, ExperimentalQuantitative
from neuroimaging.fmri.regression import fMRIRegressionOutput


canonical = neuroimaging.fmri.hrf.HRF(deriv=True)

class DelayContrast(contrast.Contrast):

    """
    Specify a delay contrast.

    Delay contrasts are specified by a sequence of functions and weights, the functions should
    NOT already be convolved with any HRF. They will be convolved with self.IRF which is expected
    to be a filter with a canonical HRF and its derivative -- defaults to the Glover model.

    If the contrast is real-valued, representing
    one column in a Formula object, then the delay for 
    this particular column is output.

    If the contrast is multi-valued, then a
    contrast of delays is computed. The contrast is formed
    by assigning a weight of 1. to each column in the function of time.


    TO DO: check that the convolved term is actually in the design column space.
    """
    def __init__(self, fn, weights, formula, IRF=canonical, name=''):
        self.IRF = IRF
        self.delayflag = True

        self.name = name
        self.formula = formula
        
        def f(namespace=None, fn=fn, time=None, n=len(fn), **extras):
            v = []
            for i in range(n):
                v.append(fn[i](namespace=namespace, time=time))
            return N.array(v)
        self.fn = f

        self.weights = N.asarray(weights)
        if len(fn) != self.weights.shape[0]:
            raise ValueError, 'length of weights does not match number of terms in DelayContrast'

        term = ExperimentalQuantitative('%s_delay' % self.name, self.fn)
        term.convolve(self.IRF)
        
        contrast.Contrast.__init__(self, term, self.formula, name=self.name)

class DelayContrastOutput(TContrastOutput):

    IRF = traits.Any()
    dt = traits.Float(0.01)
    delta = traits.ListFloat(N.linspace(-4.5,4.5,91))
    Tmax = 100.
    Tmin = -100.
    subpath = traits.Str('delays')

    def __init__(self, fmri_image, contrast, path='.', **keywords):
        traits.HasTraits.__init__(self, **keywords)
        TContrastOutput.__init__(self, fmri_image, contrast, path=path, subpath=self.subpath, **keywords)
        self.outdir = os.path.join(path, self.subpath, self.contrast.name)

    def setup_contrast(self):
        """
        Setup the contrast to output its convolution with self.IRF -- writes
        over the previous contrast's term attribute.
        """

        self.contrast.getmatrix(time=self.fmri_image.frametimes)

        self.effectmatrix = self.contrast.matrix[0::2]
        self.deltamatrix = self.contrast.matrix[1::2]

    def extract_effect(self, results):

        delay = self.contrast.IRF.delay

        self.gamma0 = N.dot(self.effectmatrix, results.beta)
        self.gamma1 = N.dot(self.deltamatrix, results.beta)

        nrow = self.gamma0.shape[0]
        self.T0sq = N.zeros(self.gamma0.shape, N.Float)
        
        for i in range(nrow):
            self.T0sq[i] = (self.gamma0[i]**2 *
                            utils.inv(results.cov_beta(matrix=self.effectmatrix[i])))

        self.r = self.gamma1 * utils.inv0(self.gamma0)
        self.rC = self.r * self.T0sq / (1. + self.T0sq)
        self.deltahat = delay.inverse(self.rC)

        self._effect = N.dot(self.contrast.weights, self.deltahat)

    def extract_sd(self, results):

        delay = self.contrast.IRF.delay

        self.T1 = N.zeros(self.gamma0.shape, N.Float)

        nrow = self.gamma0.shape[0]
        for i in range(nrow):
            self.T1[i] = self.gamma1[i] * utils.inv(N.sqrt(results.cov_beta(matrix=self.deltamatrix[i])))

        a1 = 1 + 1. * utils.inv(self.T0sq)

        gdot = N.array(([self.r * (a1 - 2.) * utils.inv0(self.gamma0 * a1**2),
                         utils.inv0(self.gamma0 * a1)] *
                        utils.inv0(delay.dforward(self.deltahat))))

        tmpcov = N.zeros((2*nrow,)*2 + self.T0sq.shape[1:], N.Float)

        Cov = results.cov_beta
        E = self.effectmatrix
        D = self.deltamatrix

        nrow = self.effectmatrix.shape[0]
            
        cov = N.zeros((nrow,)*2 + self.T0sq.shape[1:], N.Float)

        for i in range(nrow):
            for j in range(i + 1):
                cov[i,j] = (gdot[0,i] * gdot[0,j] * Cov(matrix=E[i],
                                                      other=E[j]) +  
                            gdot[0,i] * gdot[1,j] * Cov(matrix=E[i],
                                                      other=D[j]) +
                            gdot[1,i] * gdot[0,j] * Cov(matrix=D[i],
                                                      other=E[j]) +
                            gdot[1,i] * gdot[1,j] * Cov(matrix=D[i],
                                                      other=D[j]))
                cov[j,i] = cov[i,j]

        var = 0
        for i in range(nrow):
            for j in range(nrow):
                var = var + cov[i,j] * self.contrast.weights[i] * self.contrast.weights[j]

        self._sd = N.sqrt(var)                

    def extract_t(self, results):
        self._t = self._effect * utils.inv(self._sd)
        self._t = N.clip(self._t, self.Tmin, self.Tmax)

    def extract(self, results):
        self.extract_effect(results)
        self.extract_sd(results)
        self.extract_t(results)
        results = regression.ContrastResults()
        results.effect = self._effect
        results.sd = self._sd
        results.t = self._t
        return results

##     def next(self, data=None):
##         print 'here', data.t.max(), data.t.min(), data.t.mean()
##         if self.fmri_image.itervalue.type is 'slice':
##             value = copy.copy(self.fmri_image.itervalue)
##             value.slice = value.slice[1]
##         else:
##             value = self.fmri_image.itervalue

##         self.timg.next(data=data.t, value=value)
##         if self.effect:
##             self.effectimg.next(data=data.effect, value=value)
##         if self.sd:
##             self.sdimg.next(data=data.effect, value=value)

