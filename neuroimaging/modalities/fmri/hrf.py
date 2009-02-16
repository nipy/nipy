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

