"""
This module defines a class to output estimates
of delays and contrasts of delays.

Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H., Evans, A.C. (2002). \'Estimating the delay of the response in fMRI data.\' NeuroImage, 16:593-606.

"""

import neuroimaging
import copy, os, csv, string, fpformat, types
import numpy as N
import numpy.linalg as L
import enthought.traits as traits
import neuroimaging.image as image
from neuroimaging.reference import grid
from neuroimaging.fmri.regression import TContrastOutput 
from neuroimaging.statistics import utils, regression, contrast
from neuroimaging.fmri.protocol import ExperimentalRegressor, ExperimentalQuantitative
from neuroimaging.fmri.regression import fMRIRegressionOutput
from neuroimaging.statistics.regression import contrastfromcols
from neuroimaging.fmri.utils import LinearInterpolant as interpolant
from neuroimaging.fmri import hrf, filters

import pylab
from neuroimaging.fmri.plotting import MultiPlot
canplot = True
import enthought.traits as traits

class DelayContrast(contrast.Contrast):

    """
    Specify a delay contrast.

    Delay contrasts are specified by a sequence of functions and weights, the functions should
    NOT already be convolved with any HRF. They will be convolved with self.IRF which is expected
    to be a filter with a canonical HRF and its derivative -- defaults to the Glover model.

    Weights should have the same number of columns as len(fn), with each row specifying
    a different contrast.

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
            raise ValueError, 'length of weights does not match number of terms in DelayContrast'

        term = ExperimentalQuantitative('%s_delay' % self.name, self.fn)
        term.convolve(self.IRF)
        
        contrast.Contrast.__init__(self, term, self.formula, name=self.name)

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

        nout = self.contrast.weights.shape[0]

        for i in range(nout):
            rowname = self.contrast.rownames[i]
            outdir = os.path.join(self.path, self.subpath, rowname)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            cnrow = self.contrast.matrix.shape[0] / 2
            l = N.zeros(self.contrast.matrix.shape[0], N.Float)
            l[0:cnrow] = self.contrast.weights[i]

            outname = os.path.join(outdir, 't%s' % self.ext)
            timg = image.Image(outname, mode='w', grid=self.outgrid,
                               clobber=self.clobber)

            self.sync_grid(img=timg)
            self.timgs.append(timg)

            outname = os.path.join(outdir, 'effect%s' % self.ext)
            effectimg = image.Image(outname, mode='w', grid=self.outgrid,
                                    clobber=self.clobber)

            self.sync_grid(img=effectimg)
            self.effectimgs.append(effectimg)

            outname = os.path.join(outdir, 'sd%s' % self.ext)
            sdimg = iter(image.Image(outname, mode='w', grid=self.outgrid,
                                     clobber=self.clobber))

            self.sync_grid(img=sdimg)
            self.sdimgs.append(sdimg)

            matrix = N.squeeze(N.dot(l, self.contrast.matrix))

            outname = os.path.join(outdir, 'matrix%s.csv' % rowname)
            outfile = file(outname, 'w')
            outfile.write(string.join([fpformat.fix(x,4) for x in matrix], ',') + '\n')
            outfile.close()

            outname = os.path.join(outdir, 'matrix%s.bin' % rowname)
            outfile = file(outname, 'w')
            matrix = matrix.astype('<f8')
            matrix.tofile(outfile)
            outfile.close()

            if canplot:
                
                ftime = self.frametimes
                def g(time=None, **extra):
                    return N.squeeze(N.dot(l, self.contrast.term(time=time, **extra)))
                f = pylab.gcf()
                f.clf()
                pl = MultiPlot(g, tmin=0, tmax=ftime.max(),
                               dt = ftime.max() / 2000., title='Magnitude column space for delay: \'%s\'' % rowname)
                pl.draw()
                pylab.savefig(os.path.join(outdir, 'matrix%s.png' % rowname))
                f.clf()
                del(f); del(g)
                
    def extract_effect(self, results):

        delay = self.contrast.IRF.delay

        self.gamma0 = N.dot(self.effectmatrix, results.beta)
        self.gamma1 = N.dot(self.deltamatrix, results.beta)

        nrow = self.gamma0.shape[0]
        self.T0sq = N.zeros(self.gamma0.shape, N.Float)
        
        for i in range(nrow):
            self.T0sq[i] = (self.gamma0[i]**2 *
                            utils.recipr(results.cov_beta(matrix=self.effectmatrix[i])))

        self.r = self.gamma1 * utils.recipr0(self.gamma0)
        self.rC = self.r * self.T0sq / (1. + self.T0sq)
        self.deltahat = delay.inverse(self.rC)

        self._effect = N.dot(self.contrast.weights, self.deltahat)

    def extract_sd(self, results):

        delay = self.contrast.IRF.delay

        self.T1 = N.zeros(self.gamma0.shape, N.Float)

        nrow = self.gamma0.shape[0]
        for i in range(nrow):
            self.T1[i] = self.gamma1[i] * utils.recipr(N.sqrt(results.cov_beta(matrix=self.deltamatrix[i])))

        a1 = 1 + 1. * utils.recipr(self.T0sq)

        gdot = N.array(([(self.r * (a1 - 2.) *
                          utils.recipr0(self.gamma0 * a1**2)),
                         utils.recipr0(self.gamma0 * a1)] *
                        utils.recipr0(delay.dforward(self.deltahat))))

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

        nout = self.contrast.weights.shape[0]
        self._sd = N.zeros(self._effect.shape, N.Float)

        for r in range(nout):
            var = 0
            for i in range(nrow):
                var += cov[i,i] * N.power(self.contrast.weights[r,i], 2)
                for j in range(i):
                    var += 2 * cov[i,j] * self.contrast.weights[r,i] * self.contrast.weights[r,j]

            self._sd[r] = N.sqrt(var)                

    def extract_t(self, results):
        self._t = self._effect * utils.recipr(self._sd)        
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

    def next(self, data=None):
        if self.grid.itervalue.type is 'slice':
            value = copy.copy(self.grid.itervalue)
            value.slice = value.slice[1]
        else:
            value = self.grid.itervalue

        nout = self.contrast.weights.shape[0]
        
        for i in range(nout):
            self.timgs[i].next(data=data.t[i], value=value)
            if self.effect:
                self.effectimgs[i].next(data=data.effect[i], value=value)
            if self.sd:
                self.sdimgs[i].next(data=data.sd[i], value=value)

class DelayHRF(hrf.SpectralHRF):

    '''
    Delay filter with spectral or Taylor series decomposition
    for estimating delays.

    Liao et al. (2002).
    '''

    spectral = traits.true

    def __init__(self, input_hrf=hrf.canonical, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        filters.Filter.__init__(self, input_hrf)
        if self.n != 1:
            raise ValueError, 'expecting one HRF for spectral decomposition'
        self.deltaPCA()

    def deltaPCA(self, tmax=50., lower=-15.0):
        """
        Perform an expansion of fn, shifted over the values in delta. Effectively, a Taylor series approximation to fn(t+delta), in delta, with basis given by the filter elements. If fn is None, it assumes fn=IRF[0], that is the first filter.

        >>> from numpy.random import *
        >>> from BrainSTAT.fMRIstat import HRF
        >>> from pylab import *
        >>> from numpy import *
        >>>
        >>> ddelta = 0.25
        >>> delta = N.arange(-4.5,4.5+ddelta, ddelta)
        >>> time = N.arange(0,20,0.2)
        >>>
        >>> hrf = HRF.HRF(deriv=True)
        >>>
        >>> canonical = HRF.canonical
        >>> taylor = hrf.deltaPCA(delta)
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


        """

        time = N.arange(lower, tmax, self.dt)
        ntime = time.shape[0]
        irf = self.IRF

        if not self.spectral: # use Taylor series approximation

            dirf = interpolant(time, -N.gradient(irf(time), self.dt))

            H = []
            for i in range(self.delta.shape[0]):
                H.append(irf(time - self.delta[i]))
            H = N.array(H)

            W = []

            W = N.array([irf(time), dirf(time)])
            W = N.transpose(W)

            WH = N.dot(L.pinv(W), N.transpose(H))

            coef = []
            for i in range(2):
                coef.append(interpolant(self.delta, WH[i]))
            
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

        self.approx.theta, self.approx.inverse, self.approx.dinverse, self.approx.forward, self.approx.dforward = invertR(self.delta, self.approx.coef)
        
        self.delay = self.approx
        self.main = self.IRF[0]
        self.deriv = self.IRF[1]

def invertR(delta, IRF, niter=20, verbose=False):
    """
    If IRF has 2 components (w0, w1) return an estimate of the inverse of r=w1/w0, as in Liao et al. (2002). Fits a simple arctan model to the ratio w1/w0.?

    """

    R = IRF[1](delta) / IRF[0](delta)

    def f(x, theta):
        a, b, c = theta
        _x = x[:,0]
        return a * N.arctan(b * _x) + c

    def grad(x, theta):
        a, b, c = theta
        value = N.zeros((3, x.shape[0]), N.Float)
        _x = x[:,0]
        value[0] = N.arctan(b * _x)
        value[1] = a / (1. + N.power((b * _x), 2.)) * _x
        value[2] = 1.
        return N.transpose(value)

    c = delta.max() / (N.pi/2)
    n = delta.shape[0]
    delta0 = (delta[n/2+2] - delta[n/2+1])/(R[n/2+2] - R[n/2+1])
    if delta0 < 0:
        c = (delta.max() / (N.pi/2)) * 1.2
    else:
        c = -(delta.max() / (N.pi/2)) * 1.2

    from neuroimaging.statistics import nlsmodel
    design = R.reshape(R.shape[0], 1)
    model = nlsmodel.NLSModel(Y=delta,
                              design=design,
                              f=f,
                              grad=grad,
                              theta=N.array([4., 0.5, 0]),
                              niter=niter)

    for iteration in model:
        model.next()

    a, b, c = model.theta

    def _deltahat(r):
        return a * N.arctan(b * r) + c

    def _ddeltahat(r):
        return a * b / (1 + (b * r)**2) 

    def _deltahatinv(d):
        return N.tan((d - c) / a) / b

    def _ddeltahatinv(d):
        return 1. / (a * b * N.cos((d - c) / a)**2)

    for fn in [_deltahat, _ddeltahat, _deltahatinv, _ddeltahatinv]:
        setattr(fn, 'a', a)
        setattr(fn, 'b', b)
        setattr(fn, 'c', c)

    return model.theta, _deltahat, _ddeltahat, _deltahatinv, _ddeltahatinv


canonical = DelayHRF()

