"""
This module defines a class to output estimates
of delays and contrasts of delays.

Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H.,
Evans, A.C. (2002). \'Estimating the delay of the response in fMRI
data.\' NeuroImage, 16:593-606.

"""
import os, fpformat

import numpy as N
import numpy.linalg as L
from scipy.sandbox.models.utils import recipr, recipr0
from scipy.sandbox.models.contrast import Contrast, ContrastResults
from neuroimaging import traits

from neuroimaging.modalities.fmri import hrf, filters
from neuroimaging.modalities.fmri.protocol import ExperimentalQuantitative
from neuroimaging.modalities.fmri.regression import TContrastOutput 
from neuroimaging.modalities.fmri.utils import LinearInterpolant as interpolant
from neuroimaging.core.image.image import Image
from neuroimaging.modalities.fmri.fmristat.invert import invertR

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.ui.visualization.multiplot import MultiPlot

class DelayContrast(Contrast):

    """
    Specify a delay contrast.

    Delay contrasts are specified by a sequence of functions and weights, the
    functions should NOT already be convolved with any HRF. They will be
    convolved with self.IRF which is expected to be a filter with a canonical
    HRF and its derivative -- defaults to the Glover model.

    Weights should have the same number of columns as len(fn), with each row
    specifying a different contrast.

    TO DO: check that the convolved term is actually in the design column space.
    """

    def __init__(self, fn, weights, formula, IRF=None, name='', rownames=[]):
        if IRF is None:
            self.IRF = canonical
        else:
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
        if self.weights.ndim == 1:
            self.weights.shape = (1, self.weights.shape[0])

        if len(fn) != self.weights.shape[1]:
            raise ValueError, 'length of weights does not match number of ' \
                  'terms in DelayContrast'

        term = ExperimentalQuantitative('%s_delay' % self.name, self.fn)
        term.convolve(self.IRF)
        
        Contrast.__init__(self, term, self.formula, name=self.name)

        if rownames == []:
            if name == '':
                raise ValueError, 'if rownames are not specified, name must be specified'
            if self.weights.shape[0] > 1:
                self.rownames = ['%srow%d' % (name, i) for i in range(self.weights.shape[0])]
            elif self.weights.shape[0] == 1:
                self.rownames = ['']
        else:
            self.rownames = rownames

class DelayContrastOutput(TContrastOutput):

    IRF = traits.Any()
    dt = traits.Float(0.01)
    delta = traits.ListFloat(N.linspace(-4.5,4.5,91))
    Tmax = 100.
    Tmin = -100.
    subpath = traits.Str('delays')
    frametimes = traits.Any()
    
    def setup_contrast(self, time=None):
        """
        Setup the contrast for the delay.
        """

        self.contrast.getmatrix(time=self.frametimes)

        cnrow = self.contrast.matrix.shape[0] / 2
        self.effectmatrix = self.contrast.matrix[0:cnrow]
        self.deltamatrix = self.contrast.matrix[cnrow:]

    def setup_output(self):
        """
        Setup the output for contrast, the DelayContrast. One t, sd, and effect img is output for each
        row of contrast.weights. Further, the \'magnitude\' (canonical HRF) contrast matrix and \'magnitude\'
        column space are also output to illustrate what contrast this corresponds to.
        """

        self.timgs = []
        self.sdimgs = []
        self.effectimgs = []

        self.timg_iters = []
        self.sdimg_iters = []
        self.effectimg_iters = []

        nout = self.contrast.weights.shape[0]

        for i in range(nout):
            rowname = self.contrast.rownames[i]
            outdir = os.path.join(self.path, self.subpath, rowname)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            cnrow = self.contrast.matrix.shape[0] / 2
            l = N.zeros(self.contrast.matrix.shape[0])
            l[0:cnrow] = self.contrast.weights[i]

            img, it = self._setup_img(self.clobber, outdit, "t", self.ext)
            self.timgs.append(img)
            self.timg_iters.append(it)

            img, it = self._setup_img(self.clobber, outdit, "effect", self.ext)
            self.effectimgs.append(img)
            self.effectimg_iters.append(it)

            img, it = self._setup_img(self.clobber, outdit, "sd", self.ext)
            self.sdimgs.append(img)
            self.sdimg_iters.append(it)

            matrix = N.squeeze(N.dot(l, self.contrast.matrix))

            outname = os.path.join(outdir, 'matrix%s.csv' % rowname)
            outfile = file(outname, 'w')
            outfile.write(','.join(fpformat.fix(x,4) for x in matrix) + '\n')
            outfile.close()

            outname = os.path.join(outdir, 'matrix%s.bin' % rowname)
            outfile = file(outname, 'w')
            matrix = matrix.astype('<f8')
            matrix.tofile(outfile)
            outfile.close()

            if PYLAB_DEF:
                
                ftime = self.frametimes
                def g(time=None, **extra):
                    return N.squeeze(N.dot(l, self.contrast.term(time=time, **extra)))
                f = pylab.gcf()
                f.clf()
                pl = MultiPlot(g, tmin=0, tmax=ftime.max(),
                               dt = ftime.max() / 2000.,
                               title='Magnitude column space for delay: \'%s\'' % rowname)
                pl.draw()
                pylab.savefig(os.path.join(outdir, 'matrix%s.png' % rowname))
                f.clf()
                del(f); del(g)
                
    def extract_effect(self, results):

        delay = self.contrast.IRF.delay

        self.gamma0 = N.dot(self.effectmatrix, results.beta)
        self.gamma1 = N.dot(self.deltamatrix, results.beta)

        nrow = self.gamma0.shape[0]
        self.T0sq = N.zeros(self.gamma0.shape)
        
        for i in range(nrow):
            self.T0sq[i] = (self.gamma0[i]**2 *
                            recipr(results.cov_beta(matrix=self.effectmatrix[i])))

        self.r = self.gamma1 * recipr0(self.gamma0)
        self.rC = self.r * self.T0sq / (1. + self.T0sq)
        self.deltahat = delay.inverse(self.rC)

        self._effect = N.dot(self.contrast.weights, self.deltahat)

    def extract_sd(self, results):

        delay = self.contrast.IRF.delay

        self.T1 = N.zeros(self.gamma0.shape)

        nrow = self.gamma0.shape[0]
        for i in range(nrow):
            self.T1[i] = self.gamma1[i] * recipr(N.sqrt(results.cov_beta(matrix=self.deltamatrix[i])))

        a1 = 1 + 1. * recipr(self.T0sq)

        gdot = N.array(([(self.r * (a1 - 2.) *
                          recipr0(self.gamma0 * a1**2)),
                         recipr0(self.gamma0 * a1)] *
                        recipr0(delay.dforward(self.deltahat))))

        Cov = results.cov_beta
        E = self.effectmatrix
        D = self.deltamatrix

        nrow = self.effectmatrix.shape[0]
            
        cov = N.zeros((nrow,)*2 + self.T0sq.shape[1:])

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

        nout = self.contrast.weights.shape[0]
        self._sd = N.zeros(self._effect.shape)

        for r in range(nout):
            var = 0
            for i in range(nrow):
                var += cov[i,i] * N.power(self.contrast.weights[r,i], 2)
                for j in range(i):
                    var += 2 * cov[i,j] * self.contrast.weights[r,i] * self.contrast.weights[r,j]

            self._sd[r] = N.sqrt(var)                

    def extract_t(self):
        t = self._effect * recipr(self._sd)        
        t = N.clip(t, self.Tmin, self.Tmax)
        return t

    def extract(self, results):
        self.extract_effect(results)
        self.extract_sd(results)
        t = self.extract_t()

        return ContrastResults(effect=self._effect,
                               sd=self._sd,
                               t=t)

    def set_next(self, data):
        nout = self.contrast.weights.shape[0]
        for i in range(nout):
            self.timg_iters[i].set_next(data.t[i])
            if self.effect:
                self.effectimg_iters[i].set_next(data.effect[i])
            if self.sd:
                self.sdimg_iters[i].set_next(data.sd[i])

class DelayHRF(hrf.SpectralHRF):

    '''
    Delay filter with spectral or Taylor series decomposition
    for estimating delays.

    Liao et al. (2002).
    '''

    spectral = traits.true

    def __init__(self, input_hrf=hrf.canonical, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        filters.Filter.__init__(self, input_hrf, ['hrf'])
        if self.n != 1:
            raise ValueError, 'expecting one HRF for spectral decomposition'
        self.deltaPCA()

    def deltaPCA(self, tmax=50., lower=-15.0, delta=N.arange(-4.5,4.6,0.1)):
        """
        Perform an expansion of fn, shifted over the values in delta.
        Effectively, a Taylor series approximation to fn(t+delta), in delta,
        with basis given by the filter elements. If fn is None, it assumes
        fn=IRF[0], that is the first filter.

        >>> from numpy.random import *
        >>> from pylab import *
        >>> from numpy import *
        >>>
        >>> import neuroimaging.modalities.fmri.hrf as HRF
        >>> import numpy as N
        >>>
        >>> ddelta = 0.25
        >>> delta = N.arange(-4.5,4.5+ddelta, ddelta)
        >>> time = N.arange(0,20,0.2)
        >>> hrf = HRF.SpectralHRF(deriv=True)
        >>>
        >>> canonical = HRF.canonical
        >>> taylor = hrf.deltaPCA(delta=delta)
        >>> curplot = plot(time, taylor.components[1](time))
        >>> curplot = plot(time, taylor.components[0](time))
        >>> curtitle=title('Shift using Taylor series -- components')
        >>> show()
        >>>
        >>> curplot = plot(delta, taylor.coef[1](delta))
        >>> curplot = plot(delta, taylor.coef[0](delta))
        >>> curtitle = title('Shift using Taylor series -- coefficients')
        >>> show()
        >>>
        >>> curplot = plot(delta, taylor.inverse(delta))
        >>> curplot = plot(taylor.coef[1](delta) / taylor.coef[0](delta), delta)
        >>> curtitle = title('Shift using Taylor series -- inverting w1/w0')
        >>> show()
        >>>
        """

        time = N.arange(lower, tmax, self.dt)
        irf = self.IRF

        if not self.spectral: # use Taylor series approximation
            dirf = interpolant(time, -N.gradient(irf(time), self.dt))

            H = N.array([irf(time - d) for d in delta])

            W = N.array([irf(time), dirf(time)])
            W = W.T

            WH = N.dot(L.pinv(W), H.T)

            coef = [interpolant(delta, w) for w in WH]
            
            def approx(time, delta):
                value = (coef[0](delta) * irf(time)
                         + coef[1](delta) * dirf(time))
                return value

            approx.coef = coef
            approx.components = [irf, dirf]
            self.n = len(approx.components)
            self.names = [self.names[0], 'd%s' % self.names[0]]

        else:
            hrf.SpectralHRF.deltaPCA(self)

        self.approx.theta, self.approx.inverse, self.approx.dinverse, self.approx.forward, self.approx.dforward = invertR(delta, self.approx.coef)
        
        self.delay = self.approx

canonical = DelayHRF()
