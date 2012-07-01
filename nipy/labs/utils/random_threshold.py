# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
##############################################################################
# Random Thresholding Procedure (after M. Lavielle and C. Ludena)

import numpy as np
import scipy.stats as st
from nipy.algorithms.graph import wgraph_from_3d_grid
from ..group.routines import add_lines

tol = 1e-10

##############################################################################
# Wrappers


def randthresh_main(Y, K, XYZ=None, p=np.inf, varwind=False, knownull=True,
                    stop=False, verbose=False):
    """ Wrapper for random threshold functions

    Parameters
    ----------
    Y:  array of shape (n,),Observations
    K: int, Some positive integer
       (lower bound on the number of null hypotheses)
    XYZ: array of shape (3, n)  voxel coordinates.
         If not empty, connexity constraints are used on the non-null set
    p: float, optional, lp norm
    varwind: bool,
             Varying window variant (vs. fixed window, with width K)
    knownull: bool, optional,
              Known null distribution (observations assumed Exp(1) under H0)
              versus unknown (observations assumed Gaussian under H0)
    stop: bool, optional
          Stop when minimum is attained (save computation time)
    verbose: bool,  'Chatty' mode

    Returns
    -------
    D : dict
        containing the following fields:
        * "C"  (n-K) array  Lp norm of partial sums fluctuation
          about their conditional expectation
        * "thresh" <float> Detection threshold
        * "detect" (k,)    Index of detected activations

    Notes
    -----
    Random thresholding is performed only if null hypothesis of no activations
    is rejected at level 5%
    """
    if XYZ == None:
        return randthresh(Y, K, p, stop, verbose, varwind, knownull)
    else:
        return randthresh_connex(Y, K, XYZ, p, stop, verbose, varwind,
                                 knownull)


def randthresh(Y, K, p=np.inf, stop=False, verbose=False, varwind=False,
               knownull=True):
    """ Wrapper for random threshold functions (without connexity constraints)

    Parameters
    ----------
    Y: array of shape  (n,) Observations
    K: int,
       Some positive integer (lower bound on the number of null hypotheses)
    p: float, lp norm
    stop      <bool>  Stop when minimum is attained (save computation time)
    verbose   <bool>  'Chatty' mode
    varwind   <bool>  Varying window variant (vs. fixed window, with width K)
    knownull  <bool>
               Known null distribution (observations assumed Exp(1) under H0)
               versus unknown (observations assumed Gaussian under H0)
    Returns
    -------
    A dictionary D containing the following fields:
    "C" (n-K)
        Lp norm of partial sums fluctuation about their conditional expectation
    "thresh" <float> Detection threshold
    "detect" (k,)    Index of detected activations
    "v"      <float> Estimated null variance (if knownull is False)

    Notes
    -----
    Random thresholding is performed only if null hypothesis of no activations
    is rejected at level 5%
    """
    D = {}
    # Test presence of activity
    if knownull:
        X = Y
    else:
        v = np.square(Y).mean()
        X = np.clip( - np.log(1 - st.chi2.cdf(Y ** 2, 1, 0, scale=v)), 0,
                      1 / tol)
        D["v"] = v
    T = test_stat(X, p=np.inf)
    if T <= 0.65:
        print "No activity detected at 5% level"
        D["detect"] = np.array([])
        D["thresh"] = np.inf
    else:
        # Find optimal threshold
        if varwind:
            if knownull:
                C = randthresh_varwind_knownull(Y, K, p, stop, verbose)
            else:
                C, V = randthresh_varwind_gaussnull(
                    Y, K, p, stop, one_sided=False, verbose=verbose)
        else:
            if knownull:
                C = randthresh_fixwind_knownull(Y, K, p, stop, verbose)
            else:
                C, V = randthresh_fixwind_gaussnull(
                    Y, K, p, stop, one_sided=False, verbose=verbose)
        n = len(X)
        if stop:
            I = np.where(C > 0)[0]
            if len(I) > 0:
                ncoeffs = I[-1]
            else:
                ncoeffs = n - K
        else:
            I = np.where((C[2:] > C[1:-1]) * (C[1:-1] < C[:-2]))[0]
            if len(I) > 0:
                ncoeffs = I[np.argmin(C[1: -1][I])] + 1
            else:
                ncoeffs = n - K
        thresh = np.sort(np.abs(Y))[ - ncoeffs]
        # Detected activations
        detect = np.where(np.abs(Y) > thresh)[0]
        D["C"] = C[2:]
        D["thresh"] = thresh
        D["detect"] = detect
        if not knownull:
            D["v"] = V[2:]
    return D


def randthresh_connex(Y, K, XYZ, p=np.inf, stop=False, verbose=False,
                      varwind=False, knownull=True):
    """
    Wrapper for random threshold functions under connexity constraints

    Parameters
    ----------
    Y  (n,)    Observations
    K  <int>
       Some positive integer (lower bound on the number of null hypotheses)
    XYZ       (3,n)   voxel coordinates
    p         <float> lp norm
    stop      <bool>  Stop when minimum is attained (save computation time)
    verbose   <bool>  'Chatty' mode
    varwind   <bool>  Varying window variant (vs. fixed window, with width K)
    knownull  <bool>
              Known null distribution (observations assumed Exp(1) under H0)
              versus unknown (observations assumed Gaussian under H0)

    Returns
    -------
    A dictionary D containing the following fields:
    "C" (n-K)
        Lp norm of partial sums fluctuation about their conditional expectation
    "thresh"    <float>    Detection threshold
    "detect"    (ncoeffs,) Index of detected voxels

    Notes
    -----
    Random thresholding is performed only if null hypothesis of no activations
    is rejected at level 5%
    """
    # Test presence of activity
    D = {}
    if knownull:
        X = Y
    else:
        v = np.square(Y).mean()
        X = np.clip( - np.log(1 - st.chi2.cdf(Y ** 2, 1, 0, scale=v)),
                       0, 1 / tol)
        D["v"] = v
    T = test_stat(X, p=np.inf)
    if T <= 0.65:
        print "No activity detected at 5% level"
        D["detect"] = np.array([])
        D["thresh"] = np.inf
    else:
        # Find optimal threshold
        if varwind:
            if knownull:
                C = randthresh_varwind_knownull_connex(
                    Y, K, XYZ, p, stop, verbose)
            else:
                C, V = randthresh_varwind_gaussnull_connex(
                    Y, K, XYZ, p, stop, verbose=verbose)
        else:
            if knownull:
                C = randthresh_fixwind_knownull_connex(
                    Y, K, XYZ, p, stop, verbose)
            else:
                C, V = randthresh_fixwind_gaussnull_connex(
                    Y, K, XYZ, p, stop, verbose=verbose)
        n = len(X)
        if stop:
            I = np.where(C > 0)[0]
            if len(I) > 0:
                ncoeffs = I[-1]
            else:
                ncoeffs = n - K
        else:
            I = np.where((C[2:] > C[1:-1]) * (C[1:-1] < C[:-2]))[0]
            if len(I) > 0:
                ncoeffs = I[np.argmin(C[1:-1][I])] + 1
            else:
                ncoeffs = n - K
        thresh = np.sort(np.abs(Y))[ - ncoeffs]
        detect = np.where(np.abs(Y) > thresh)[0]

        # Remove isolated voxels
        iso = isolated(XYZ[:, detect])
        detect[iso] = -1
        detect = detect[detect != -1]
        D["C"] = C[2:]
        D["thresh"] = thresh
        D["detect"] = detect
        if knownull == False:
            Ynull = np.square(Y).copy()
            Ynull[detect] = np.nan
            Ynull = Ynull[np.isnan(Ynull) == False]
            D["v"] = V[2:]
    return D

#########################################################################
# random threshold functions without connexity constraints


def randthresh_fixwind_knownull(X, K, p=np.inf, stop=False, verbose=False):
    """Random threshold with fixed-window and known null distribution

    Parameters
    ==========
    X (n,): Observations (must be Exp(1) under H0)
    K <int>:
       Some positive integer (lower bound on the number of null hypotheses)
    p <float>: Lp norm
    stop <bool>: Stop when minimum is attained (save computation time)

    Returns
    =======
    C (n-K):
      Lp norm of partial sums fluctuation about their conditional expectation
    """
    n = len(X)
    I = 1.0 / np.arange(1, n + 1)
    # Sort data
    sortX = np.sort(X)[:: - 1]
    C = np.zeros(n - K, float)
    T = np.cumsum(sortX)
    for k in xrange(2, n - K):
        #Ratio of expectations
        B = np.arange(1, K + 1) * (1 + I[:n - 1 - k].sum() - I[:K].cumsum())
        B /= float(K) * ( 1 + I[K: n - 1 - k].sum() )
        #Partial sums
        Tk = T[k + 1: k + K + 1] - T[k]
        #Conditional expectations
        Q = B * Tk[-1]
        if p == np.inf:
            C[k] = np.abs(Tk - Q).max() / np.sqrt(n)
        else:
            C[k] = ( np.abs(Tk - Q) ** p ).sum() / n ** (p / 2.0 + 1)
        if verbose:
            print "k :", k, "C[k]:", C[k]
        if C[k] > C[k - 1] and C[k - 1] < C[k - 2] and stop:
            break
    return C


def randthresh_varwind_knownull(X, K, p=np.inf, stop=False, verbose=False):
    """Random threshold with varying window and known null distribution

    Parameters
    ==========
    X (n,): Observations (Exp(1) under H0)
    K <int>:
      Some positive integer (lower bound on the number of null hypotheses)
    p <float>: lp norm
    stop <bool>: Stop when minimum is attained (save computation time)

    Returns
    =======
    C (n-K)
      Lp norm of partial sums fluctuation about their conditional expectation
    """
    n = len(X)
    I = 1.0 / np.arange(1, n + 1)
    #Sort data
    sortX = np.sort(X)[:: - 1]
    T = np.cumsum(sortX)
    C = np.zeros(n - K, float)
    for k in xrange(2, n - K):
        #Ratio of expectations
        B = np.arange(1, n - k) * ( 1 + I[:n - 1 - k].sum() -
                                    I[:n - k - 1].cumsum())
        B /= float(n - k - 1)
        #Partial sums
        Tk = T[k + 1:] - T[k]
        #Conditional expectations
        Q = B * Tk[ - 1]
        if p == np.inf:
            C[k] = np.abs(Tk - Q).max() / np.sqrt(n - k - 1)
        else:
            C[k] = ( np.abs(Tk - Q) ** p).sum() / (n - k - 1) ** (p / 2.0 + 1)
        if verbose:
            print "k:", k, "C[k]:", C[k]
        if C[k] > C[k - 1] and C[k - 1] < C[k - 2] and stop:
            break
    return C


def randthresh_fixwind_gaussnull(Y, K, p=np.inf, stop=False, one_sided=False,
                                 verbose=False):
    """ Random threshold with fixed window and null gaussian distribution

    Parameters
    ==========
    Y array of shape (n,)
      Observations (assumed Gaussian under H0, with unknown variance)
    K, int,  Some positive integer
       (lower bound on the number of null hypotheses)
    p, float, lp norm
    stop: bool,
          Stop when minimum is attained (save computation time)
    one_sided: bool,
               If nonzero means are positive only (vs. positive or negative)

    Returns
    =======
    C array of shape (n-K)
      Lp norm of partial sums fluctuation about their conditional expectation
    """
    n = len(Y)
    I = 1.0 / np.arange(1, n + 1)
    if one_sided:
        sortY = np.sort(Y)
        std = np.sqrt((np.sum(sortY[1:K] ** 2) + np.cumsum(sortY[K: n] ** 2))\
                          * 1.0 / np.arange(K, n))
        std = std[:: - 1]
    else:
        sortY = np.sort(np.square(Y))
        V = (np.sum(sortY[1: K]) + np.cumsum(sortY[K: n])) * \
            1.0 / np.arange(K, n)
        V = V[:: - 1]
    C = np.zeros(n - K, float)
    sortY = sortY[:: - 1]
    for k in xrange(2, n - K):
        if one_sided:
            X = np.clip( - np.log(1 - st.norm.cdf(sortY[k + 1: k + K + 1],
                                                  scale=std[k])), 0, 1 / tol)
        else:
            X = np.clip( -
                np.log(1 - st.chi2.cdf(sortY[k + 1: k + K + 1],
                                       1, 0, scale=V[k])), 0, 1 / tol)

        # Ratio of expectations
        B = np.arange(1, K + 1) * (1 + I[:n - 1 - k].sum() - I[: K].cumsum())
        B /= float(K) * (1 + I[K: n - 1 - k].sum())
        # Partial sums
        T = X.cumsum()
        # Conditional expectations
        Q = B * T[-1]
        if p == np.inf:
            C[k] = np.abs(T - Q).max() / np.sqrt(n)
        else:
            C[k] = ( np.abs(T - Q) ** p ).sum() / n ** ( p / 2.0 + 1)
        if verbose:
            print "k:", k, "C[k]:", C[k]
        if C[k] > C[k-1] and C[k-1] < C[k-2] and stop:
            break
    return C, V


def randthresh_varwind_gaussnull(Y, K, p=np.inf, stop=False, one_sided=False,
                                 verbose=False):
    """Random threshold with fixed window and gaussian null distribution

    Parameters
    ==========
    Y  (n,) Observations (assumed Gaussian under H0, with unknown variance)
    K  <int>
       Some positive integer (lower bound on the number of null hypotheses)
    p         <float> lp norm
    stop      <bool>  Stop when minimum is attained (save computation time)
    one_sided <bool>
              If nonzero means are positive only (vs. positive or negative)

    Returns
    =======
    C (n-K)
       Lp norm of partial sums fluctuation about their conditional expectation
    """
    n = len(Y)
    I = 1.0 / np.arange(1, n + 1)
    if one_sided:
        sortY = np.sort(Y)
        std = np.sqrt((np.sum(sortY[1: K] ** 2) + np.cumsum(sortY[K: n] ** 2))
                      * 1.0 / np.arange(K, n))
        std = std[:: - 1]
    else:
        sortY = np.sort(np.square(Y))
        V = (np.sum(sortY[1: K]) + np.cumsum(sortY[K: n]))\
            * 1.0 / np.arange(K, n)
        V = V[:: - 1]
    C = np.zeros(n - K, float)
    sortY = sortY[:: - 1]
    for k in xrange(2, n - K):
        if one_sided:
            X = np.clip( - np.log(1 - st.norm.cdf(sortY[k + 1:],
                                                  scale=std[k])), 0, 1 / tol)
        else:
            X = np.clip( -
                np.log(1 - st.chi2.cdf(sortY[k + 1:], 1, 0, scale=V[k])), 0,
                1 / tol)
        # Ratio of expectations
        B = np.arange(1, n - k) * ( 1 + I[: n - 1 - k].sum() - \
                                    I[: n - k - 1].cumsum() )
        B /= float(n - k - 1)
        # Partial sums
        T = X.cumsum()
        # Conditional expectations
        Q = B * T[ - 1]
        if p == np.inf:
            C[k] = np.abs(T - Q).max() / np.sqrt(n)
        else:
            C[k] = ( np.abs(T - Q) ** p ).sum() / n ** (p / 2.0 + 1)
        if verbose:
            print "k:", k, "C[k]:", C[k]
        if C[k] > C[k - 1] and C[k - 1] < C[k - 2] and stop:
            break
    return C, V

###############################################################################
# random threshold functions with connexity constraints


def randthresh_fixwind_knownull_connex(X, K, XYZ, p=np.inf, stop=False,
                                       verbose=False):
    """Random threshold with fixed-window and known null distribution,
    using connexity constraint on non-null set.

    Parameters
    ==========
    X (n,): Observations (must be Exp(1) under H0)
    XYZ (3,n): voxel coordinates
    K <int>:
      Some positive integer (lower bound on the number of null hypotheses)
    p <float>: Lp norm
    stop <bool>: Stop when minimum is attained (save computation time)

    Returns
    =======
    C (n-K):
      Lp norm of partial sums fluctuation about their conditional expectation
    """
    n = len(X)
    I = 1.0 / np.arange(1, n + 1)
    #Sort data
    J = np.argsort(X)[:: - 1]
    sortX = X[J]
    C = np.zeros(n - K, float)
    T = np.zeros(K, float)
    L = np.zeros(n, int)
    L[J[0]] = 1
    for k in xrange(2, n - K):
        #Ratio of expectations
        B = np.arange(1, K + 1) * (1 + I[: n - 1 - k].sum() - I[: K].cumsum())
        B /= float(K) * ( 1 + I[K: n - 1 - k].sum() )
        Jk = J[:k]
        #Suprathreshold voxels connected to new voxel
        XYZk = np.abs(XYZ[:, Jk] - XYZ[:, J[k - 1]].reshape(3, 1))
        Lk = np.where((XYZk.sum(axis=0) <= 2) * (XYZk.max(axis=0) <= 1))[0]\
            [: - 1]
        if len(Lk) == 0:
            L[J[k - 1]] = 1
        else:
            L[J[Lk]] = 0
        Ik = np.where(L[Jk] == 1)[0]
        nk = len(Ik)
        #Partial sums
        if nk >= K:
            T = sortX[Ik[:K]].cumsum()
        elif nk == 0:
            T = sortX[k + 1: k + K + 1].cumsum()
        else:
            T[:nk] = sortX[Ik].cumsum()
            T[nk:] = T[nk - 1] + sortX[k + 1:k + K - nk + 1].cumsum()
        # Conditional expectations
        Q = B * T[-1]
        if p == np.inf:
            C[k] = np.abs(T - Q).max() / np.sqrt(n)
        else:
            C[k] = (np.abs(T - Q) ** p).sum() / n ** (p / 2.0 + 1)
        if verbose:
            print "k:", k, "nk:", nk, "C[k]:", C[k]
        if C[k] > C[k - 1] and C[k - 1] < C[k - 2] and stop:
            break
    return C


def randthresh_varwind_knownull_connex(X, K, XYZ, p=np.inf, stop=False,
                                       verbose=False):
    """Random threshold with varying window and known null distribution

    Parameters
    ==========
    X (n,): Observations (Exp(1) under H0)
    K <int>:
      Some positive integer (lower bound on the number of null hypotheses)
    XYZ (3,n): voxel coordinates
    p <float>: lp norm
    stop <bool>: Stop when minimum is attained (save computation time)

    Returns
    =======
    C (n-K)
      Lp norm of partial sums fluctuation about their conditional expectation
    """
    n = len(X)
    I = 1.0 / np.arange(1, n + 1)
    #Sort data
    J = np.argsort(X)[:: - 1]
    sortX = X[J]
    C = np.zeros(n - K, float)
    L = np.zeros(n, int)
    L[J[0]] = 1
    for k in xrange(2, n - K):
        Jk = J[:k]
        #Suprathreshold voxels connected to new voxel
        XYZk = np.abs(XYZ[:, Jk] - XYZ[:, J[k-1]].reshape(3, 1))
        Lk = np.where((XYZk.sum(axis=0) <= 2) * (XYZk.max(axis=0) <= 1))\
            [0][:-1]
        if len(Lk) == 0:
            L[J[k - 1]] = 1
        else:
            L[J[Lk]] = 0
        Ik = np.where(L[Jk] == 1)[0]
        #Ik = isolated(XYZ[:, Jk])
        nk = len(Ik)
        #Ratio of expectations
        B = np.arange(1, n - k + nk) * ( 1 + I[:n - 1 - k + nk].sum() -
                                         I[:n - k - 1 + nk].cumsum())
        B /= float(n - k - 1 + nk)
        #Partial sums
        if nk == 0:
            T = sortX[k + 1:].cumsum()
        else:
            T = np.zeros(n - k + nk - 1, float)
            T[:nk] = sortX[Ik].cumsum()
            T[nk:] = T[nk - 1] + sortX[k + 1:].cumsum()

        #Conditional expectations
        Q = B * T[-1]
        if p == np.inf:
            C[k] = np.abs(T - Q).max() / np.sqrt(n - k - 1 + nk)
        else:
            C[k] = ( np.abs(T - Q) ** p ).sum() / (n - k - 1 + nk) **\
                (p / 2.0 + 1)
        if C[k] > C[k - 1] and C[k - 1] < C[k - 2] and stop:
            break
        if verbose:
            print "k:", k, "nk:", nk, "C[k]:", C[k]
    return C


def randthresh_fixwind_gaussnull_connex(X, K, XYZ, p=np.inf, stop=False,
                                        verbose=False):
    """Random threshold with fixed-window and gaussian null distribution,
    using connexity constraint on non-null set.

    Parameters
    ==========
    X (n,): Observations (assumed Gaussian under H0)
    XYZ (3,n): voxel coordinates
    K <int>:
      Some positive integer (lower bound on the number of null hypotheses)
    p <float>: Lp norm
    stop <bool>: Stop when minimum is attained (save computation time)

    Returns
    =======
    C (n-K):
       Lp norm of partial sums fluctuation about their conditional expectation
    """
    n = len(X)
    I = 1.0 / np.arange(1, n + 1)
    #Sort data
    J = np.argsort(X ** 2)[:: - 1]
    sortX = np.square(X)[J]
    C = np.zeros(n - K, float)
    V = np.zeros(n - K, float)
    T = np.zeros(K, float)
    L = np.zeros(n, int)
    L[J[0]] = 1
    for k in xrange(2, n - K):
        #Ratio of expectations
        B = np.arange(1, K + 1) * ( 1 + I[:n - 1 - k].sum() - I[:K].cumsum())
        B /= float(K) * ( 1 + I[K:n - 1 - k].sum())
        Jk = J[:k]
        #Suprathreshold voxels connected to new voxel
        XYZk = np.abs(XYZ[:, Jk] - XYZ[:, J[k - 1]].reshape(3, 1))
        Lk = np.where((XYZk.sum(axis=0) <= 2) *
                      (XYZk.max(axis=0) <= 1))[0][: - 1]
        if len(Lk) == 0:
            L[J[k - 1]] = 1
        else:
            L[J[Lk]] = 0
        Ik = np.where(L[Jk] == 1)[0]
        nk = len(Ik)
        #Null variance
        V[k] = (sortX[Ik].sum() + sortX[k + 1:].sum()) / float(nk + n - k - 1)
        #Partial sums
        if nk >= K:
            T = np.clip( - np.log(1 - st.chi2.cdf(
                        sortX[Ik[:K]], 1, 0, scale=V[k])), 0, 1 / tol).cumsum()
        elif nk == 0:
            T = np.clip( -
                np.log(1 - st.chi2.cdf(sortX[k + 1:k + K + 1], 1, 0,
                                       scale=V[k])), 0, 1 / tol).cumsum()
        else:
            T[:nk] = np.clip( -
                np.log(1 - st.chi2.cdf(sortX[Ik], 1, 0, scale=V[k])), 0,
                1 / tol).cumsum()
            T[nk:] = T[nk - 1] + np.clip( -
                np.log(1 - st.chi2.cdf(sortX[k + 1:k + K - nk + 1], 1, 0,
                                       scale=V[k])), 0, 1 / tol).cumsum()
        # Conditional expectations
        Q = B * T[ - 1]
        if p == np.inf:
            C[k] = np.abs(T - Q).max() / np.sqrt(n)
        else:
            C[k] = (np.abs(T - Q) ** p).sum() / n ** (p / 2.0 + 1)
        if verbose:
            print "k:", k, "nk:", nk, "C[k]:", C[k]
        if C[k] > C[k - 1] and C[k - 1] < C[k - 2] and stop:
            break
    return C, V


def randthresh_varwind_gaussnull_connex(X, K, XYZ, p=np.inf, stop=False,
                                        verbose=False):
    """Random threshold with fixed-window and gaussian null distribution,
    using connexity constraint on non-null set.

    Parameters
    ==========
    X (n,): Observations (assumed Gaussian under H0)
    XYZ (3,n): voxel coordinates
    K <int>:
      Some positive integer (lower bound on the number of null hypotheses)
    p <float>: Lp norm
    stop <bool>: Stop when minimum is attained (save computation time)

    Returns
    =======
    C (n-K):
      Lp norm of partial sums fluctuation about their conditional expectation
    """
    n = len(X)
    I = 1.0 / np.arange(1, n + 1)
    #Sort data
    J = np.argsort(X ** 2)[:: - 1]
    sortX = np.square(X)[J]
    C = np.zeros(n - K, float)
    V = np.zeros(n - K, float)
    T = np.zeros(K, float)
    L = np.zeros(n, int)
    L[J[0]] = 1
    for k in xrange(2, n - K):
        Jk = J[:k]
        #Suprathreshold voxels connected to new voxel
        XYZk = np.abs(XYZ[:, Jk] - XYZ[:, J[k - 1]].reshape(3, 1))
        Lk = np.where((XYZk.sum(axis=0) <= 2) *
                      (XYZk.max(axis=0) <= 1))[0][: - 1]
        if len(Lk) == 0:
            L[J[k - 1]] = 1
        else:
            L[J[Lk]] = 0
        Ik = np.where(L[Jk] == 1)[0]
        #Ik = isolated(XYZ[:, Jk])
        nk = len(Ik)
        #Ratio of expectations
        B = np.arange(1, n - k + nk) * ( 1 + I[:n - 1 - k + nk].sum() -
                                         I[:n - k - 1 + nk].cumsum())
        B /= float(n - k - 1 + nk)
        #Null variance
        V[k] = (sortX[Ik].sum() + sortX[k + 1:].sum()) / float(nk + n - k - 1)
        #Partial sums
        if nk == 0:
            T = np.clip( -
                np.log(1 - st.chi2.cdf(sortX[k + 1:], 1, 0, scale=V[k])), 0,
                1 / tol).cumsum()
        else:
            T = np.zeros(n - k + nk - 1, float)
            T[:nk] = np.clip( -
                np.log(1 - st.chi2.cdf(sortX[Ik], 1, 0, scale=V[k])), 0,
                1 / tol).cumsum()
            T[nk:] = T[nk-1] + np.clip( -
                np.log(1 - st.chi2.cdf(sortX[k + 1:], 1, 0, scale=V[k])), 0,
                1 / tol).cumsum()
        #Conditional expectations
        Q = B * T[-1]
        if p == np.inf:
            C[k] = np.abs(T - Q).max() / np.sqrt(n - k - 1 + nk)
        else:
            C[k] = ( np.abs(T - Q) ** p).sum() / \
                (n - k - 1 + nk) ** (p / 2.0 + 1)
        if verbose:
            print "k:", k, "nk:", nk, "C[k]:", C[k]
        if C[k] > C[k - 1] and C[k - 1] < C[k - 2] and stop:
            break
    return C, V

#############################################################################
# Miscellanous functions


def test_stat(X, p=np.inf):
    """Test statistic of global null hypothesis
    that all observations have zero-mean

    Parameters
    ==========
    X (n,)    : X[j] = -log(1-F(|Y[j]|))
                where F: cdf of |Y[j]| under null hypothesis
                (must be computed beforehand)
    p         : Lp norm (<= inf) to use for computing test statistic

    Returns
    =======
    D <float> : test statistic
    """
    n = len(X)
    I = 1.0 / np.arange(1, n + 1)
    #Partial sums
    T = np.cumsum(np.sort(X)[:: - 1])
    #Expectation of partial sums
    E = np.arange(1, n + 1) * (1 + I.sum() - I.cumsum())
    #Conditional expectation of partial sums
    Q = E / n * T[ - 1]
    #Test statistic
    if p == np.inf:
        return np.max( ( np.abs(T - Q) ) / np.sqrt(n) )
    else:
        return sum(np.abs(T - Q) ** p) / (n ** (0.5 * p + 1))


def isolated(XYZ, k=18):
    """
    Outputs an index I of isolated points from their integer coordinates,
    XYZ (3, n), and under k-connectivity, k = 6, 18 or 26.
    """
    label = wgraph_from_3d_grid(XYZ.T, k).cc()
    # Isolated points
    ncc = label.max() + 1
    p = XYZ.shape[1]
    size = np.zeros(ncc, float)
    ones = np.ones((p, 1), float)
    add_lines(ones, size.reshape(ncc, 1), label)
    return np.where(size[label] == 1)[0]
