# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" This module provides definitions of various hemodynamic response
functions (hrf).

In particular, it provides Gary Glover's canonical HRF, AFNI's default
HRF, and a spectral HRF.
"""

import numpy as np
from sympy import Symbol, DeferredVector, exp, Derivative, abs
from .formula import aliased_function
from .utils import lambdify_t, T

# Sympy symbols used below

deft = DeferredVector('t')

def gamma_params(peak_location, peak_fwhm):
    """ Parameters for gamma density given peak and width
    
    TODO: where does the coef come from again.... check fmristat code

    From a peak location and peak fwhm, determine the parameters of a
    Gamma density

    f(x) = coef * x**(alpha-1) * exp(-x*beta)

    The coefficient returned ensures that the f has integral 1 over
    [0,np.inf]

    Parameters
    ----------
    peak_location : float
       Location of the peak of the Gamma density
    peak_fwhm : float
       FWHM at the peak

    Returns
    -------
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
    return alpha, beta, coef


def gamma_expr(peak_location, peak_fwhm):
    alpha, beta, coef = gamma_params(peak_location, peak_fwhm)
    return coef * ((T >= 0) * (T+1.0e-14))**(alpha) * exp(-(T+1.0e-14)/beta)


# Glover canonical HRF models
# they are both Sympy objects

def _getint(f, dt=0.02, t=50):
    # numerical integral of function
    lf = lambdify_t(f)
    tt = np.arange(dt,t+dt,dt)
    return lf(tt).sum() * dt 

deft = DeferredVector('t')
_gexpr = gamma_expr(5.4, 5.2) - 0.35 * gamma_expr(10.8,7.35)
_gexpr = _gexpr / _getint(_gexpr)
_glover = lambdify_t(_gexpr)
glover = aliased_function('glover', _glover)
n = {}
glovert = lambdify_t(glover(deft))

# Derivative of Glover HRF

_dgexpr = _gexpr.diff(T)
dpos = Derivative((T >= 0), T)
_dgexpr = _dgexpr.subs(dpos, 0)
_dgexpr = _dgexpr / _getint(abs(_dgexpr))
_dglover = lambdify_t(_dgexpr)
dglover = aliased_function('dglover', _dglover)
dglovert = lambdify_t(dglover(deft))

del(_glover); del(_gexpr); del(dpos); del(_dgexpr); del(_dglover)

# AFNI's HRF

_aexpr = ((T >= 0) * T)**8.6 * exp(-T/0.547)
_aexpr = _aexpr / _getint(_aexpr)
_afni = lambdify_t(_aexpr)
afni = aliased_function('afni', _afni)
afnit = lambdify_t(afni(deft))


# Primitive of the HRF -- temoprary fix to handle blocks
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
    alpha = np.power(peak_location / peak_fwhm, 2) * 8 * np.log(2.0)
    beta = np.power(peak_fwhm, 2) / peak_location / 8 / np.log(2.0)
    ak = int(np.round(alpha+1))
    P = np.sum([1./sp.gamma(k+1)*((T/beta)**k) for k in range(ak)],0)
    return (T > 0) * (1-exp(-T/beta)*P)

_igexpr = igamma_params(5.4, 5.2) - 0.35 * igamma_params(10.8,7.35)
_igexpr = _igexpr / _getint(_igexpr)
_iglover = lambdify_t(_igexpr)
iglover = aliased_function('iglover', _iglover)
iglovert = lambdify_t(iglover(deft))

