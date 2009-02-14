"""
This module provides definitions of various hemodynamic response functions
(hrf).

In particular, it provides Gary Glover's canonical HRF, AFNI's default HRF, and
a spectral HRF.
"""

__docformat__ = 'restructuredtext'


import numpy as np
import numpy.linalg as L
from sympy import Symbol, lambdify, DeferredVector, exp, Derivative

from neuroimaging.modalities.fmri import filters
from neuroimaging.modalities.fmri.fmristat.invert import invertR

# Sympy symbols used below

vector_t = DeferredVector('vt')
t = Symbol('t')


def gamma_params(peak_location, peak_fwhm):
    """
    TODO: where does the coef come from again.... check fmristat code

    From a peak location and peak fwhm,
    determine the parameters of a Gamma density

    f(x) = coef * x**(alpha-1) * exp(-x*beta)

    The coefficient returned ensures that
    the f has integral 1 over [0,np.inf]

    :Parameters:
        peak_location : float
            Location of the peak of the Gamma density
        peak_fwhm : float
            FWHM at the peak

    :Returns: 
        alpha : float
            Shape parameter in the Gamma density
        beta : float
            Scale parameter in the Gamma density
        coef : float
            Coefficient needed to ensure the density has integral 1.
    """
    alpha = np.power(peak_location / peak_fwhm, 2) * 8 * np.log(2.0)
    beta = np.power(peak_fwhm, 2) / peak_location / 8 / np.log(2.0)
    coef = peak_location**(-alpha) * np.exp(peak_location / beta)
    return coef * ((t > 0) * t)**(alpha-1) * exp(-beta*t) 

# Glover canonical HRF models
# they are both Sympy objects

def vectorize_time(f):
    """
    Take a sympy expression that contains the symbol 't'
    and return a lambda with a vectorized time.
    """
    return lambdify(vector_t, f.subs(t, vector_t), 'numpy')

def _getint(f, dt=0.02, t=50):
    lf = vectorize_time(f)
    tt = np.arange(dt,t+dt,dt)
    return lf(tt).sum() * dt 

glover_sympy = gamma_params(5.4, 5.2) - 0.35 * gamma_params(10.8,7.35)
glover_sympy = glover_sympy / _getint(glover_sympy)

dglover_sympy = glover_sympy.diff(t)

dpos = Derivative((t > 0), t)
dglover_sympy = dglover_sympy.subs(dpos, 0)
dglover_sympy = dglover_sympy / _getint(dglover_sympy)

# This is callable

glover = vectorize_time(glover_sympy)
glover.__doc__ = """
Canonical HRF
"""
canonical = glover #TODO :get rid of 'canonical'

dglover = vectorize_time(dglover_sympy)
dglover.__doc__ = """
Derivative of canonical HRF
"""

# AFNI's default HRF (at least at some point in the past)

afni_sympy = ((t > 0) * t)**8.6 * exp(-t/0.547)
afni_sympy =  afni_sympy / _getint(afni_sympy)
afni = vectorize_time(afni_sympy)

class SpectralHRF(filters.Filter):
    '''
    Delay filter with spectral or Taylor series decomposition
    for estimating delays.

    Liao et al. (2002).
    '''

    def __init__(self, input_hrf=glover, spectral=True, ncomp=2,
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
        H = np.nan_to_num(np.asarray(H))
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


