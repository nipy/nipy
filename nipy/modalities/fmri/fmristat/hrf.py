"""
This module computes the canonical HRF used in 
fMRIstat, both the 2-term spectral approximation
and the Taylor series approximation, to a shifted
version of the canonical Glover HRF.

References
==========

Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H.,
    Evans, A.C. (2002). \'Estimating the delay of the response in fMRI
    data.\' NeuroImage, 16:593-606.
"""

import numpy as np
import numpy.linalg as L
from scipy.interpolate import interp1d

from sympy import Function
from nipy.modalities.fmri import hrf, formula
from nipy.modalities.fmri.fmristat.invert import invertR

def spectral_decomposition(hrf2decompose, ncomp=2, tmax=50, tmin=-15, dt=0.02,
                           delta=np.arange(-4.5, 4.6, 0.1)):
    """

    Perform a PCA expansion of fn, shifted over the values in delta,
    returning the first ncomp components.
    This smooths out the HRF as compared to using a Taylor series
    approximation.

    Parameters
    ==========

    hrf2decompose : sympy expression 
        An expression that can be vectorized
        as a function of 't'. This is the HRF to be expanded in PCA

    ncomp : int
        Number of principal components to retain.

    """

    hrft = hrf.vectorize(hrf2decompose(hrf.t))
    time = np.arange(tmin, tmax, dt)

    H = []
    for i in range(delta.shape[0]):
        H.append(hrft(time - delta[i]))
    H = np.nan_to_num(np.asarray(H))
    U, S, V = L.svd(H.T, full_matrices=0)

    basis = []
    for i in range(ncomp):
        b = interp1d(time, U[:, i], bounds_error=False, fill_value=0.)
        if i == 0:
            d = np.fabs((b(time) * dt).sum())
        b.y /= d
        basis.append(b)

    W = np.array([b(time) for b in basis[:ncomp]])
    WH = np.dot(L.pinv(W.T), H.T)
    coef = [interp1d(delta, w, bounds_error=False, fill_value=0.) for w in WH]
            
    if coef[0](0) < 0:
        coef[0].y *= -1.
        basis[0].y *= -1.

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
        symbasis.append(formula.aliased_function('%s%d' % (str(hrf), i), b))
    return symbasis, approx


def taylor_approx(hrf2decompose, tmax=50, tmin=-15, dt=0.02,
                  delta=np.arange(-4.5, 4.6, 0.1)):
    """

    Perform a PCA expansion of fn, shifted over the values in delta,
    returning the first ncomp components.
    This smooths out the HRF as compared to using a Taylor series
    approximation.

    Parameters
    ----------

    hrf2decompose : sympy expression 
        An expression that can be vectorized
        as a function of 't'. This is the HRF to be expanded in PCA

    ncomp : int
        Number of principal components to retain.

    References
    ----------

    Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H.,
    Evans, A.C. (2002). \'Estimating the delay of the response in fMRI
    data.\' NeuroImage, 16:593-606.

    """

    hrft = hrf.vectorize(hrf2decompose(hrf.t))
    time = np.arange(tmin, tmax, dt)

    dhrft = interp1d(time, -np.gradient(hrft(time), dt), bounds_error=False,
                    fill_value=0.)

    dhrft.y *= 2
    H = np.array([hrft(time - d) for d in delta])
    W = np.array([hrft(time), dhrft(time)])
    W = W.T

    WH = np.dot(L.pinv(W), H.T)

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
