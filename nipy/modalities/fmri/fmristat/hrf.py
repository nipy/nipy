# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Computation of the canonical HRF used in fMRIstat, both the 2-term
spectral approximation and the Taylor series approximation, to a shifted
version of the canonical Glover HRF.

References
----------
Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H.,
    Evans, A.C. (2002). \'Estimating the delay of the response in fMRI
    data.\' NeuroImage, 16:593-606.
"""
from __future__ import absolute_import

import numpy as np
import numpy.linalg as npl

from sympy.utilities.lambdify import implemented_function

from ..utils import T, lambdify_t, Interp1dNumeric
from .. import hrf
from .invert import invertR

def spectral_decomposition(hrf2decompose,
                           time=None,
                           delta=None,
                           ncomp=2):
    """ PCA decomposition of symbolic HRF shifted over time

    Perform a PCA expansion of a symbolic HRF, time shifted over the
    values in delta, returning the first ncomp components.

    This smooths out the HRF as compared to using a Taylor series
    approximation.

    Parameters
    ----------
    hrf2decompose : sympy expression
        An expression that can be lambdified as a function of 't'. This
        is the HRF to be expanded in PCA
    time : None or np.ndarray, optional
        None gives default value of np.linspace(-15,50,3251) chosen to
        match fMRIstat implementation.  This corresponds to a time
        interval of 0.02.  Presumed to be equally spaced.
    delta : None or np.ndarray, optional
        None results in default value of np.arange(-4.5, 4.6, 0.1)
        chosen to match fMRIstat implementation.
    ncomp : int, optional
        Number of principal components to retain.

    Returns
    -------
    hrf : [sympy expressions]
        A sequence length `ncomp` of symbolic HRFs that are the
        principal components.
    approx : 
        TODO
    """
    if time is None:
        time = np.linspace(-15,50,3251)
    dt = time[1] - time[0]
    if delta is None:
        delta = np.arange(-4.5, 4.6, 0.1)
    # make numerical implementation from hrf function and symbol t.
    # hrft returns function values when called with values for time as
    # input.
    hrft = lambdify_t(hrf2decompose(T))
    # Create stack of time-shifted HRFs.  Time varies over row, delta
    # over column.
    ts_hrf_vals = np.array([hrft(time - d) for d in delta]).T
    ts_hrf_vals = np.nan_to_num(ts_hrf_vals)
    # PCA 
    U, S, V = npl.svd(ts_hrf_vals, full_matrices=0)
    # make interpolators from the generated bases
    basis = []
    for i in range(ncomp):
        b = Interp1dNumeric(time, U[:, i], bounds_error=False, fill_value=0.)
        # normalize components witn integral of abs of first component
        if i == 0: 
            d = np.fabs((b(time) * dt).sum())
        b.y /= d
        basis.append(b)
    # reconstruct time courses for all bases
    W = np.array([b(time) for b in basis]).T
    # regress basis time courses against original time shifted time
    # courses, ncomps by len(delta) parameter matrix
    WH = np.dot(npl.pinv(W), ts_hrf_vals)
    # put these into interpolators to get estimated coefficients for any
    # value of delta
    coef = [Interp1dNumeric(delta, w, bounds_error=False, fill_value=0.)
            for w in WH]
    # swap sign of first component to match that of input HRF.  Swap
    # other components if we swap the first, to standardize signs of
    # components across SVD implementations.
    if coef[0](0) < 0: # coefficient at time shift of 0
        for i in range(ncomp):
            coef[i].y *= -1.
            basis[i].y *= -1.

    def approx(time, delta):
        value = 0
        for i in range(ncomp):
            value += coef[i](delta) * basis[i](time)
        return value

    approx.coef = coef
    approx.components = basis
    (approx.theta,
     approx.inverse,
     approx.dinverse,
     approx.forward,
     approx.dforward) = invertR(delta, approx.coef)
    # construct aliased functions from bases
    symbasis = []
    for i, b in enumerate(basis):
        symbasis.append(
            implemented_function('%s%d' % (str(hrf2decompose), i), b))
    return symbasis, approx


def taylor_approx(hrf2decompose,
                  time=None,
                  delta=None):
    """ A Taylor series approximation of an HRF shifted by times `delta`

    Returns original HRF and gradient of HRF

    Parameters
    ----------
    hrf2decompose : sympy expression
        An expression that can be lambdified as a function of 't'. This
        is the HRF to be expanded in PCA
    time : None or np.ndarray, optional
        None gives default value of np.linspace(-15,50,3251) chosen to
        match fMRIstat implementation.  This corresponds to a time
        interval of 0.02.  Presumed to be equally spaced.
    delta : None or np.ndarray, optional
        None results in default value of np.arange(-4.5, 4.6, 0.1)
        chosen to match fMRIstat implementation.

    Returns
    -------
    hrf : [sympy expressions]
        Sequence length 2 comprising (`hrf2decompose`, ``dhrf``) where
        ``dhrf`` is the first derivative of `hrf2decompose`.
    approx : 
        TODO

    References
    ----------
    Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H.,
    Evans, A.C. (2002). \'Estimating the delay of the response in fMRI
    data.\' NeuroImage, 16:593-606.
    """
    if time is None:
        time = np.linspace(-15,50,3251)
    dt = time[1] - time[0]
    if delta is None:
        delta = np.arange(-4.5, 4.6, 0.1)
    # make numerical implementation from hrf function and symbol t.
    # hrft returns function values when called with values for time as
    # input.
    hrft = lambdify_t(hrf2decompose(T))
    # interpolator for negative gradient of hrf
    dhrft = Interp1dNumeric(time, -np.gradient(hrft(time), dt),
                            bounds_error=False, fill_value=0.)
    dhrft.y *= 2
    # Create stack of time-shifted HRFs.  Time varies over row, delta
    # over column.
    ts_hrf_vals = np.array([hrft(time - d) for d in delta]).T
    # hrf, dhrf
    W = np.array([hrft(time), dhrft(time)]).T
    # regress hrf, dhrf at times against stack of time-shifted hrfs
    WH = np.dot(npl.pinv(W), ts_hrf_vals)
    # put these into interpolators to get estimated coefficients for any
    # value of delta
    coef = [Interp1dNumeric(delta, w, bounds_error=False,
                            fill_value=0.) for w in WH]
            
    def approx(time, delta):
        value = (coef[0](delta) * hrft(time)
                 + coef[1](delta) * dhrft(time))
        return value

    approx.coef = coef
    approx.components = [hrft, dhrft]
    (approx.theta,
     approx.inverse,
     approx.dinverse,
     approx.forward,
     approx.dforward) = invertR(delta, approx.coef)
    dhrf = implemented_function('d%s' % str(hrf2decompose), dhrft)
    return [hrf2decompose, dhrf], approx


canonical, canonical_approx = taylor_approx(hrf.glover)
spectral, spectral_approx = spectral_decomposition(hrf.glover)
