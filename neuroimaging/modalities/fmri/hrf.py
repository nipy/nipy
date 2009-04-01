"""
This module provides definitions of various hemodynamic response functions
(hrf).

In particular, it provides Gary Glover's canonical HRF, AFNI's default HRF, and
a spectral HRF.
"""

__docformat__ = 'restructuredtext'


import numpy as np
import numpy.linalg as L
from sympy import Symbol, lambdify, DeferredVector, exp, Derivative, Integral, uppergamma
from formula import Term
from utils import Vectorize, t


from neuroimaging.modalities.fmri.fmristat.invert import invertR

# Sympy symbols used below

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
    t = Term('t')
    alpha = np.power(peak_location / peak_fwhm, 2) * 8 * np.log(2.0)
    beta = np.power(peak_fwhm, 2) / peak_location / 8 / np.log(2.0)
    coef = peak_location**(-alpha) * np.exp(peak_location / beta)
    return coef * ((t > 1.e-14) * (t+1.e-14))**(alpha) * exp(-t/beta)

def igamma_params(peak_location, peak_fwhm):
    """
    From a peak location and peak fwhm,
    determine the paramteres of a Gamma density
    and return an approximate (accurate) approximation of its integral
    f(x) = int_0^x  coef * t**(alpha-1) * exp(-t*beta) dt
    so that lim_{x->infty} f(x)=1
    
    :Parameters:
        peak_location : float
            Location of the peak of the Gamma density
        peak_fwhm : float
            FWHM at the peak

    :Returns:
         the function of t

    NOTE: this is only a temporary fix,
    and will have to be removed in the long term
    """
    import scipy.special as sp
    t = Term('t')
    alpha = np.power(peak_location / peak_fwhm, 2) * 8 * np.log(2.0)
    beta = np.power(peak_fwhm, 2) / peak_location / 8 / np.log(2.0)
    #coef = peak_location**(-alpha) * np.exp(peak_location / beta)
    #return uppergamma(alpha+1,t/beta)
    # fixme: approximation
    #return (t > 0) * (1-exp(-t/(beta)))
    ak = int(np.round(alpha+1))
    P = np.sum([1./sp.gamma(k+1)*((t/beta)**k) for k in range(ak)],0)
    return (t > 0) * (1-exp(-t/beta)*P)
    
    
# Glover canonical HRF models
# they are both Sympy objects

def _getint(f, dt=0.02, t=50):
    lf = Vectorize(f)
    tt = np.arange(dt,t+dt,dt)
    return np.sqrt((lf(tt)**2).sum() * dt )

class Subs(object):

    def __call__(self, s):
        return self._expr.subs(t, s)


class Glover(Subs):

    _expr = gamma_params(5.4, 5.2) - 0.35 * gamma_params(10.8,7.35)
    _expr = _expr / _getint(_expr)

glover_sympy = Glover()
glover = Vectorize(glover_sympy._expr)

class DGlover(Subs):

    dglover_sympy = Glover._expr.diff(t)
    dpos = Derivative((t > 0), t)
    dglover_sympy = dglover_sympy.subs(dpos, 0)
    _expr = dglover_sympy / _getint(dglover_sympy)
    del(dglover_sympy); del(dpos)

dglover_sympy = DGlover()
dglover = Vectorize(dglover_sympy._expr)


# not the following is here only temporarily to handle blocks
class IGlover(Subs):    
    _expr = igamma_params(5.4, 5.2) - 0.35 * igamma_params(10.8,7.35)

iglover_sympy = IGlover()
iglover = Vectorize(iglover_sympy._expr)


class AFNI(Subs):
    # AFNI's default HRF (at least at some point in the past)

    _expr = ((t > 0) * t)**8.6 * exp(-t/0.547)
    _expr = _expr / _getint(_expr)

afni_sympy =  AFNI()
afni = Vectorize(afni_sympy._expr)

