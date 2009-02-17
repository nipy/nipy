"""
This module provides definitions of various hemodynamic response functions
(hrf).

In particular, it provides Gary Glover's canonical HRF, AFNI's default HRF, and
a spectral HRF.
"""

__docformat__ = 'restructuredtext'


import numpy as np
import numpy.linalg as L

from neuroimaging.modalities.fmri import filters
from neuroimaging.modalities.fmri.utils import LinearInterpolant as interpolant
from neuroimaging.modalities.fmri.fmristat.invert import invertR

def glover2GammaDENS(peak_hrf, fwhm_hrf):
    """
    :Parameters:
        `peak_hfr` : TODO
            TODO
        `fwhm_hrf` : TODO
            TODO
    
    :Returns: TODO
    """
    alpha = np.power(peak_hrf / fwhm_hrf, 2) * 8 * np.log(2.0)
    beta = np.power(fwhm_hrf, 2) / peak_hrf / 8 / np.log(2.0)
    coef = peak_hrf**(-alpha) * np.exp(peak_hrf / beta)
    return coef, filters.GammaDENS(alpha + 1., 1. / beta)

def _glover(peak_hrf=(5.4, 10.8), fwhm_hrf=(5.2, 7.35), dip=0.35):
    coef1, gamma1 = glover2GammaDENS(peak_hrf[0], fwhm_hrf[0])
    coef2, gamma2 = glover2GammaDENS(peak_hrf[1], fwhm_hrf[1])
    f = filters.GammaCOMB([[coef1, gamma1], [-dip*coef2, gamma2]])
    dt = 0.02
    t = np.arange(0, 50 + dt, dt)
    c = (f(t) * dt).sum()
    return filters.GammaCOMB([[coef1/c, gamma1], [-dip*coef2/c, gamma2]])

# Glover, 'canonical HRF'

glover = filters.Filter(_glover(), names=['glover'])
glover_deriv = filters.Filter([_glover(), _glover().deriv(const=-1.)],
                              names=['glover', 'dglover'])
canonical = glover


# AFNI's default HRF (at least at some point in the past)

afni = filters.Filter(filters.GammaDENS(9.6, 1.0/0.547), ['gamma'])

class SpectralHRF(filters.Filter):
    '''
    Delay filter with spectral or Taylor series decomposition
    for estimating delays.

    Liao et al. (2002).
    '''

    def __init__(self, input_hrf=canonical, spectral=True, ncomp=2,
                 names=['glover'], deriv=False, **keywords):
        """
        :Parameters:
            `input_hrf` : TODO
                TODO
            `spectral` : bool
                TODO
            `ncomp` : int
                TODO
            `names` : TODO
                TODO
            `deriv` : bool
                TODO
            `keywords` : dict
                passed as keyword arguments to `filters.Filter.__init__`
        """
        filters.Filter.__init__(self, input_hrf, names=names, **keywords)
        self.deriv = deriv
        self.ncomp = ncomp
        self.spectral = spectral
        if self.n != 1:
            raise ValueError, 'expecting one HRF for spectral decomposition'
        self.deltaPCA()

    def deltaPCA(self, tmax=50., lower=-15.0, delta=None):
        """
        Perform an expansion of fn, shifted over the values in delta.
        Effectively, a Taylor series approximation to fn(t+delta), in delta,
        with basis given by the filter elements. If fn is None, it assumes
        fn=IRF[0], that is the first filter.

        >>> GUI = True
        >>> import numpy as np
        >>> from pylab import plot, title, show
        >>> from neuroimaging.modalities.fmri.hrf import glover, glover_deriv, SpectralHRF
        >>>
        >>> ddelta = 0.25
        >>> delta = np.arange(-4.5,4.5+ddelta, ddelta)
        >>> time = np.arange(0,20,0.2)
        >>>
        >>> hrf = SpectralHRF(glover)
        >>>
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
        """

        if delta is None: delta = np.arange(-4.5, 4.6, 0.1)
        time = np.arange(lower, tmax, self.dt)
        if callable(self.IRF):
            irf = self.IRF
        else:
            irf = self.IRF[0]

        H = []
        for i in range(delta.shape[0]):
            H.append(irf(time - delta[i]))
        H = np.array(H)


        U, S, V = L.svd(H.T, full_matrices=0)

        basis = []
        for i in range(self.ncomp):
            b = interpolant(time, U[:, i])

            if i == 0:
                d = np.fabs((b(time) * self.dt).sum())
            b.f.y /= d
            basis.append(b)


        W = np.array([b(time) for b in basis[:self.ncomp]])

        WH = np.dot(L.pinv(W.T), H.T)
        
        coef = [interpolant(delta, w) for w in WH]
            
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

        (self.approx.theta,
         self.approx.inverse,
         self.approx.dinverse,
         self.approx.forward,
         self.approx.dforward) = invertR(delta, self.approx.coef)
        return approx


