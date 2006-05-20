import numpy as N
import numpy.linalg as L
import filters
from enthought import traits
from neuroimaging.fmri.utils import LinearInterpolant as interpolant

def glover2GammaDENS(peak_hrf, fwhm_hrf):
    alpha = N.power(peak_hrf / fwhm_hrf, 2) * 8 * N.log(2.0)
    beta = N.power(fwhm_hrf, 2) / peak_hrf / 8 / N.log(2.0)
    coef = peak_hrf**(-alpha) * N.exp(peak_hrf / beta)
    return coef, filters.GammaDENS(alpha + 1., 1. / beta)

def _glover(peak_hrf=[5.4, 10.8], fwhm_hrf=[5.2, 7.35], dip=0.35):
    coef1, gamma1 = glover2GammaDENS(peak_hrf[0], fwhm_hrf[0])
    coef2, gamma2 = glover2GammaDENS(peak_hrf[1], fwhm_hrf[1])
    f = filters.GammaCOMB([[coef1,gamma1],[-dip*coef2, gamma2]])
    dt = 0.02
    t = N.arange(0, 50 + dt, dt)
    c = (f(t) * dt).sum()
    return filters.GammaCOMB([[coef1/c,gamma1],[-dip*coef2/c, gamma2]])

# Glover, 'canonical HRF'

glover = filters.Filter(_glover(), names=['glover'])
glover_deriv = filters.Filter([_glover(), _glover().deriv(const=-1.)],
                              names=['glover', 'dglover'])
canonical = glover

# AFNI's default HRF (at least at some point in the past)

afni = filters.Filter(filters.GammaDENS(9.6, 1.0/0.547))

class SpectralHRF(filters.Filter):
    dt = traits.Float(0.02)
    tmax = traits.Float(500.0)
    ncomp = traits.Int(2)
    names = traits.ListStr(['glover'])

    '''
    Delay filter with spectral or Taylor series decomposition
    for estimating delays.

    Liao et al. (2002).
    '''

    spectral = traits.true

    def __init__(self, input_hrf=canonical, **keywords):
        filters.Filter.__init__(self, input_hrf, **keywords)
        if self.n != 1:
            raise ValueError, 'expecting one HRF for spectral decomposition'
        self.deltaPCA()

    def deltaPCA(self, tmax=50., lower=-15.0, delta=N.arange(-4.5,4.6,0.1)):
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

        H = []
        for i in range(self.delta.shape[0]):
            H.append(irf(time - self.delta[i]))
        H = N.array(H)

            
        U, S, V = L.svd(N.transpose(H), full_matrices=0)
        prcnt_var_spectral = N.sum(S[0:self.ncomp]**2) / N.sum(S**2) * 100

        basis = []
        for i in range(self.ncomp):
            b = interpolant(time, U[:,i])
            
            if i == 0:
                b_int = (b(time) * self.dt).sum()
                b.f.y /= N.fabs(b_int)
            else:
                b_int = (N.fabs(b(time)) * self.dt).sum()
                b.f.y /= b_int
            basis.append(b)

        W = []
        for i in range(self.ncomp):
            W.append(basis[i](time))
        W = N.transpose(W)

        WH = N.dot(L.pinv(W), N.transpose(H))
        
        coef = []
        for i in range(self.ncomp):
            coef.append(interpolant(self.delta, WH[i]))
            
        if coef[0](0) < 0:
            coef[0].f.y *= -1.
            basis[0].f.y *= -1.

        def approx(time, delta):
            value = 0
            for i in range(self.ncomp):
                value += coef[i](delta) * basis[i](time)
            return value

        approx.coef = coef
        approx.components = basis

        self.approx = approx
        self.IRF = approx.components
        self.n = len(approx.components)
        self.names = ['%s_%d' % (self.names[0], i) for i in range(self.n)]

        if self.n == 1:
            self.IRF = self.IRF[0]
