# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" This module provides definitions of various hemodynamic response
functions (hrf).

In particular, it provides Gary Glover's canonical HRF, AFNI's default
HRF, and a spectral HRF.

The Glover HRF is based on:

@article{glover1999deconvolution,
  title={{Deconvolution of impulse response in event-related BOLD fMRI}},
  author={Glover, G.H.},
  journal={NeuroImage},
  volume={9},
  number={4},
  pages={416--429},
  year={1999},
  publisher={Orlando, FL: Academic Press, c1992-}
}

This paramaterization is from fmristat:

http://www.math.mcgill.ca/keith/fmristat/

fmristat models the HRF as the difference of two gamma functions, ``g1``
and ``g2``, each defined by the timing of the gamma function peaks
(``pk1, pk2``) and the fwhms (``width1, width2``):

   raw_hrf = g1(pk1, width1) - a2 * g2(pk2, width2)

where ``a2`` is the scale factor for the ``g2`` gamma function.  The
actual hrf is the raw hrf set to have an integral of 1. 

fmristat used ``pk1, width1, pk2, width2, a2 = (5.4 5.2 10.8 7.35
0.35)``.  These are parameters to match Glover's 1 second duration
auditory stimulus curves.  Glover wrote these as:

   y(t) = c1 * t**n1 * exp(t/t1) - a2 * c2 * t**n2 * exp(t/t2)

with ``n1, t1, n2, t2, a2 = (6.0, 0.9, 12, 0.9, 0.35)``.  The difference
between Glover's expression and ours is because we (and fmristat) use
the peak location and width to characterize the function rather than
``n1, t1``.  The values we use are equivalent.  Specifically, in our
formulation:

>>> n1, t1, c1 = gamma_params(5.4, 5.2)
>>> np.allclose((n1-1, t1), (6.0, 0.9), rtol=0.02)
True
>>> n2, t2, c2 = gamma_params(10.8, 7.35)
>>> np.allclose((n2-1, t2), (12.0, 0.9), rtol=0.02)
True
"""

import numpy as np
import sympy
# backwards compatibility with sympy 0.6.x
try:
    sympy_abs = sympy.Abs # 0.7.0
except AttributeError:
    sympy_abs = sympy.abs

from nipy.fixes.sympy.utilities.lambdify import implemented_function

from .utils import lambdify_t, T

def gamma_params(peak_location, peak_fwhm):
    """ Parameters for gamma density given peak and width
    
    TODO: where does the coef come from again.... check fmristat code

    From a peak location and peak fwhm, determine the parameters (shape,
    scale) of a Gamma density:

    f(x) = coef * x**(shape-1) * exp(-x/scale)

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
    shape : float
       Shape parameter in the Gamma density
    scale : float
       Scale parameter in the Gamma density
    coef : float
       Coefficient needed to ensure the density has integral 1.
    """
    shape_m1 = np.power(peak_location / peak_fwhm, 2) * 8 * np.log(2.0)
    scale = np.power(peak_fwhm, 2) / peak_location / 8 / np.log(2.0)
    coef = peak_location**(-shape_m1) * np.exp(peak_location / scale)
    return shape_m1 + 1, scale, coef


def gamma_expr(peak_location, peak_fwhm):
    shape, scale, coef = gamma_params(peak_location, peak_fwhm)
    return (
        coef * ((T >= 0) * (T+1.0e-14))**(shape-1)
        * sympy.exp(-(T+1.0e-14)/scale)
        )


# Glover canonical HRF models
# they are both Sympy objects

def _getint(f, dt=0.02, t=50):
    # numerical integral of function
    lf = lambdify_t(f)
    tt = np.arange(dt,t+dt,dt)
    return lf(tt).sum() * dt 


_gexpr = gamma_expr(5.4, 5.2) - 0.35 * gamma_expr(10.8, 7.35)
_gexpr = _gexpr / _getint(_gexpr)
_glover = lambdify_t(_gexpr)
glover = implemented_function('glover', _glover)
glovert = lambdify_t(glover(T))

# Derivative of Glover HRF

_dgexpr = _gexpr.diff(T)
dpos = sympy.Derivative((T >= 0), T)
_dgexpr = _dgexpr.subs(dpos, 0)
_dgexpr = _dgexpr / _getint(sympy_abs(_dgexpr))
_dglover = lambdify_t(_dgexpr)
dglover = implemented_function('dglover', _dglover)
dglovert = lambdify_t(dglover(T))

del(_glover); del(_gexpr); del(dpos); del(_dgexpr); del(_dglover)

# AFNI's HRF

_aexpr = ((T >= 0) * T)**8.6 * sympy.exp(-T/0.547)
_aexpr = _aexpr / _getint(_aexpr)
_afni = lambdify_t(_aexpr)
afni = implemented_function('afni', _afni)
afnit = lambdify_t(afni(T))
