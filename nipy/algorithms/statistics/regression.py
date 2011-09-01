# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module provides various convenience functions for extracting
statistics from regression analysis techniques to model the
relationship between the dependent and independent variables.

As well as a convenience class to output the result, RegressionOutput

"""

__docformat__ = 'restructuredtext'

import numpy as np
import numpy.linalg as L
from scipy.linalg import toeplitz
from ..utils.matrices import pos_recipr

def output_T(contrast, results, effect=None, sd=None, t=None):
    """ Convenience function to collect t contrast results

    Parameters
    ----------
    contrast : array
        contrast matrix
    results : object
        implementing Tcontrast method
    effect : {None, False, True}, optional
        flag as to whether to append effect (None -> False)
    sd : {None, False, True}, optional
        flag as to whether to append standard deviation (None -> False)
    t : {None, False, True}, optional
        flag as to whether to append t (None -> False)

    Returns
    -------
    res_list : list
        List of results.  It will be len(3) if all of `effect`, `sd` and `t` are
        True, and empty if they are all False
    """
    store = []
    if effect:
        store.append('effect')
    if sd:
        store.append('sd')
    if t:
        store.append('t')
    r = results.Tcontrast(contrast, store=store)
    v = []
    if effect:
        v.append(r.effect)
    if sd:
        v.append(r.sd)
    if t:
        v.append(r.t)
    return v


def output_F(contrast, results):
    """
    This convenience function outputs the results of an Fcontrast
    from a regression
    """
    return results.Fcontrast(contrast).F


def output_resid(results):
    """
    This convenience function outputs the residuals
    from a regression
    """
    return results.resid


class RegressionOutput(object):
    """
    A class to output things in GLM passes through arrays of data.
    """

    def __init__(self, img, fn, output_shape=None):
        """
        Parameters
        ----------
        img : ``Image`` instance
            The output Image
        fn : callable
            A function that is applied to a
            models.model.LikelihoodModelResults instance
        """
        self.img = img
        self.fn = fn
        self.output_shape = output_shape

    def __call__(self, x):
        return self.fn(x)

    def __setitem__(self, index, value):
        self.img[index] = value


class RegressionOutputList(object):
    """
    A class to output more than one thing
    from a GLM pass through arrays of data.
    """

    def __call__(self, x):
        return self.fn(x)

    def __init__(self, imgs, fn):
        """ Initialize regression output list

        Parameters
        ----------
        imgs : list
            The list of output images
        fn : callable
            A function that is applied to a
            models.model.LikelihoodModelResults instance
        """
        self.list = imgs
        self.fn = fn

    def __setitem__(self, index, value):
        self.list[index[0]][index[1:]] = value


class TOutput(RegressionOutputList):
    """
    Output contrast related to a T contrast
    from a GLM pass through data.
    """
    def __init__(self, contrast, effect=None,
                 sd=None, t=None):
        self.fn = lambda x: output_T(contrast,
                                     x,
                                     effect=effect,
                                     sd=sd,
                                     t=t)
        self.list = []
        if effect is not None:
            self.list.append(effect)
        if sd is not None:
            self.list.append(sd)
        if t is not None:
            self.list.append(t)


class ArrayOutput(RegressionOutput):
    """
    Output an array from a GLM pass through data.

    By default, the function called is output_resid, so residuals
    are output.
    """
    def __init__(self, img, fn):
        RegressionOutput.__init__(self, img, fn)


def output_AR1(results):
    """
    Compute the usual AR(1) parameter on
    the residuals from a regression.
    """
    resid = results.resid
    rho = np.add.reduce(resid[0:-1]*resid[1:] / np.add.reduce(resid[1:-1]**2))
    return rho


class AREstimator(object):
    """
    A class that whose instances can estimate
    AR(p) coefficients from residuals
    """
    def __init__(self, model, p=1):
        """ Bias-correcting AR estimation class

        Parameters
        ----------
        model : ``OSLModel`` instance
            A models.regression.OLSmodel instance, where `model` has attribute ``design``
        p : int, optional
            Order of AR(p) noise
        """
        self.p = p
        self._setup_bias_correct(model)

    def _setup_bias_correct(self, model):
        R = np.identity(model.design.shape[0]) - np.dot(model.design, model.calc_beta)
        M = np.zeros((self.p+1,)*2)
        I = np.identity(R.shape[0])
        for i in range(self.p+1):
            Di = np.dot(R, toeplitz(I[i]))
            for j in range(self.p+1):
                Dj = np.dot(R, toeplitz(I[j]))
                M[i,j] = np.diagonal((np.dot(Di, Dj))/(1.+(i>0))).sum()
        self.invM = L.inv(M)
        return

    def __call__(self, results):
        """ Calculate AR(p) coefficients from `results`.``residuals``

        Parameters
        ----------
        results : Results instance
            A models.model.LikelihoodModelResults instance

        Returns
        -------
        ar_p : array
            AR(p) coefficients
        """
        resid = results.resid.reshape(
            (results.resid.shape[0], np.product(results.resid.shape[1:])))
        sum_sq = results.scale.reshape(resid.shape[1:]) * results.df_resid
        cov = np.zeros((self.p + 1,) + sum_sq.shape)
        cov[0] = sum_sq
        for i in range(1, self.p+1):
            cov[i] = np.add.reduce(resid[i:] * resid[0:-i], 0)
        cov = np.dot(self.invM, cov)
        output = cov[1:] * pos_recipr(cov[0])
        return np.squeeze(output)
