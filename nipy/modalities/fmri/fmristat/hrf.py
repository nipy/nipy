""" Computation of the canonical HRF used in fMRIstat, both the 2-term
spectral approximation and the Taylor series approximation, to a shifted
version of the canonical Glover HRF.

References
----------
Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H.,
    Evans, A.C. (2002). \'Estimating the delay of the response in fMRI
    data.\' NeuroImage, 16:593-606.
"""

import numpy as np
import numpy.linalg as npl
from scipy.interpolate import interp1d

from sympy import Function
from nipy.modalities.fmri import hrf, formula
from nipy.modalities.fmri.fmristat.invert import invertR


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
        An expression that can be vectorized
        as a function of 't'. This is the HRF to be expanded in PCA
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
    hrft = hrf.vectorize(hrf2decompose(hrf.t))
    H = []
    for i in range(delta.shape[0]):
        H.append(hrft(time - delta[i]))
    H = np.nan_to_num(np.asarray(H))
    U, S, V = npl.svd(H.T, full_matrices=0)
    basis = []
    for i in range(ncomp):
        b = interp1d(time, U[:, i], bounds_error=False, fill_value=0.)
        if i == 0:
            d = np.fabs((b(time) * dt).sum())
        b.y /= d
        basis.append(b)
    W = np.array([b(time) for b in basis[:ncomp]])
    WH = np.dot(npl.pinv(W.T), H.T)
    coef = [interp1d(delta, w, bounds_error=False, fill_value=0.) for w in WH]
    # swap sign of first component to match that of input HRF.  Swap
    # other components if we swap the first to standardize signs of
    # components across SVD implementations.
    if coef[0](0) < 0:
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
    symbasis = []
    for i, b in enumerate(basis):
        symbasis.append(
            formula.aliased_function('%s%d' % (str(hrf2decompose), i), b))
    return symbasis, approx


def taylor_approx(hrf2decompose,
                  time=None,
                  delta=None):
    """ A Taylor series approximation of an HRF shifted by times `delta`

    Returns the HRF and the first derivative.

    Parameters
    ----------
    hrf2decompose : sympy expression
        An expression that can be vectorized as a function of 't'. 
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
    hrft = hrf.vectorize(hrf2decompose(hrf.t))
    dhrft = interp1d(time, -np.gradient(hrft(time), dt), bounds_error=False,
                    fill_value=0.)
    dhrft.y *= 2
    H = np.array([hrft(time - d) for d in delta])
    W = np.array([hrft(time), dhrft(time)])
    W = W.T
    WH = np.dot(npl.pinv(W), H.T)
    coef = [interp1d(delta, w, bounds_error=False,
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
    dhrf = formula.aliased_function('d%s' % str(hrf2decompose), dhrft)
    return [hrf2decompose, dhrf], approx


canonical, canonical_approx = taylor_approx(hrf.glover)
spectral, spectral_approx = spectral_decomposition(hrf.glover)
