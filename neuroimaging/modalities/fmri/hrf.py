"""
This module provides definitions of various hemodynamic response functions
(hrf).

In particular, it provides Gary Glover's canonical HRF, AFNI's default HRF, and
a spectral HRF.
"""

__docformat__ = 'restructuredtext'


import numpy as np
from sympy import Symbol, lambdify, DeferredVector, exp, Derivative, abs, Function
from formula import Term, vectorize, add_aliases_to_namespace, t, aliased_function

# Sympy symbols used below

def gamma_params(peak_location, peak_fwhm):
    """
    TODO: where does the coef come from again.... check fmristat code

    From a peak location and peak fwhm,
    determine the parameters of a Gamma density

    f(x) = coef * x**(alpha-1) * exp(-x*beta)

    The coefficient returned ensures that
    the f has integral 1 over [0,np.inf]

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
    return coef * ((t >= 0) * (t+1.0e-14))**(alpha) * exp(-(t+1.0e-14)/beta)

# Glover canonical HRF models
# they are both Sympy objects

def _getint(f, dt=0.02, t=50):
    lf = vectorize(f)
    tt = np.arange(dt,t+dt,dt)
    return lf(tt).sum() * dt 

deft = DeferredVector('t')
_gexpr = gamma_params(5.4, 5.2) - 0.35 * gamma_params(10.8,7.35)
_gexpr = _gexpr / _getint(_gexpr)
_glover = vectorize(_gexpr)
glover = aliased_function('glover', _glover)
n = {}
glovert = lambdify(deft, glover(deft), add_aliases_to_namespace(glover, n))

# Derivative of Glover HRF

_dgexpr = _gexpr.diff(t)
dpos = Derivative((t >= 0), t)
_dgexpr = _dgexpr.subs(dpos, 0)
_dgexpr = _dgexpr / _getint(abs(_dgexpr))
_dglover = vectorize(_dgexpr)
dglover = aliased_function('dglover', _dglover)
dglovert = lambdify(deft, dglover(deft), add_aliases_to_namespace(dglover, n))

del(_glover); del(_gexpr); del(dpos); del(_dgexpr); del(_dglover)

# AFNI's HRF

_aexpr = ((t >= 0) * t)**8.6 * exp(-t/0.547)
_aexpr = _aexpr / _getint(_aexpr)
_afni = lambdify(deft, _aexpr.subs(t, deft), 'numpy')
afni = aliased_function('afni', _afni)
del(_afni)
afnit = lambdify(deft, afni(deft), add_aliases_to_namespace(afni, n))

