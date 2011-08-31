# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for one sample t-tests
"""
__docformat__ = 'restructuredtext'

import numpy as np
from ..utils.matrices import pos_recipr

def estimate_mean(Y, sd):
    """ Estimate the mean of a sample given information about
    the standard deviations of each entry.

    Parameters
    ----------
    Y : ndarray
       Data for which mean is to be estimated.  Should have shape[0] ==
       number of subjects.
    sd : ndarray
       Standard deviation (subject specific) of the data for which the
       mean is to be estimated.  Should have shape[0] == number of
       subjects.

    Returns
    -------
    value : dict
       This dictionary has keys ['effect', 'scale', 't', 'resid', 'sd']
    """
    nsubject = Y.shape[0]
    squeeze = False
    if Y.ndim == 1:
        Y = Y.reshape(Y.shape[0], 1)
        squeeze = True

    _stretch = lambda x: np.multiply.outer(np.ones(nsubject), x)

    W = pos_recipr(sd**2)
    if W.shape in [(), (1,)]:
        W = np.ones(Y.shape) * W
    W.shape = Y.shape

    # Compute the mean using the optimal weights
    effect = (Y * W).sum(0) / W.sum(0)
    resid = (Y - _stretch(effect)) * np.sqrt(W)

    scale = np.add.reduce(np.power(resid, 2), 0) / (nsubject - 1)
    var_total = scale * pos_recipr(W.sum(0))

    value = {}
    value['resid'] = resid
    value['effect'] = effect
    value['sd'] = np.sqrt(var_total)
    value['t'] = value['effect'] * pos_recipr(value['sd'])
    value['scale'] = np.sqrt(scale)

    if squeeze:
        for key in value.keys():
            value[key] = np.squeeze(value[key])
    return value

def estimate_varatio(Y, sd, df=None, niter=10):
    """ Estimate variance fixed/random effects variance ratio

    In a one-sample random effects problem, estimate
    the ratio between the fixed effects variance and
    the random effects variance.

    Parameters
    ----------

    Y : np.ndarray
        Data for which mean is to be estimated.
        Should have shape[0] == number of subjects.
    sd : array
        Standard deviation (subject specific) 
        of the data for which the mean is to be estimated.
        Should have shape[0] == number of subjects.
    df : int or None, optional
        If supplied, these are used as weights when
        deriving the fixed effects variance. Should have
        length == number of subjects.
    niter : int, optional
        Number of EM iterations to perform (default 10)

    Returns
    -------
    value : dict
       This dictionary has keys ['fixed', 'ratio', 'random'], where
       'fixed' is the fixed effects variance implied by the input
       parameter 'sd'; 'random' is the random effects variance and
       'ratio' is the estimated ratio of variances: 'random'/'fixed'.
    """
    nsubject = Y.shape[0]
    squeeze = False
    if Y.ndim == 1:
        Y = Y.reshape(Y.shape[0], 1)
        squeeze = True
    _stretch = lambda x: np.multiply.outer(np.ones(nsubject), x)

    W = pos_recipr(sd**2)
    if W.shape in [(), (1,)]:
        W = np.ones(Y.shape) * W
    W.shape = Y.shape

    S = 1. / W
    R = Y - np.multiply.outer(np.ones(Y.shape[0]), Y.mean(0))
    sigma2 = np.squeeze((R**2).sum(0)) / (nsubject - 1)

    Sreduction = 0.99
    minS = S.min(0) * Sreduction
    Sm = S - _stretch(minS)

    for _ in range(niter):
        Sms = Sm + _stretch(sigma2)
        W = pos_recipr(Sms)
        Winv = pos_recipr(W.sum(0))
        mu = Winv * (W*Y).sum(0)
        R = W * (Y - _stretch(mu))
        ptrS = 1 + (Sm * W).sum(0) - (Sm * W**2).sum(0) * Winv
        sigma2 = np.squeeze((sigma2 * ptrS + (sigma2**2) * (R**2).sum(0)) / nsubject)
    sigma2 = sigma2 - minS
    if df is None:
        df = np.ones(nsubject)
    df.shape = (1, nsubject)
    _Sshape = S.shape
    S.shape = (S.shape[0], np.product(S.shape[1:]))

    value = {}
    value['fixed'] = (np.dot(df, S) / df.sum()).reshape(_Sshape[1:])
    value['ratio'] = np.nan_to_num(sigma2 / value['fixed'])
    value['random'] = sigma2

    if squeeze:
        for key in value.keys():
            value[key] = np.squeeze(value[key])
    return value
