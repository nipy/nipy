# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Convenience functions and classes for statistics on images.

These functions and classes support the return of statistical test results from
iterations through data.

The basic container here is the RegressionOutput.  This does two basic things:

* via __call__, processes a result object from a regression to produce
  something, usually an array
* via slicing (__setitem__), it can store stuff, usually arrays.

We use these by other objects (see algorithms.statistics.fmri.fmristat) slicing
data out of images, fitting models to the data to create results objects, and
then passing them to these here ``RegressionOutput`` containers via call, to get
useful arrays, and then putting the results back into the ``RegressionOutput``
containers via slicing (__setitem__).
"""

__docformat__ = 'restructuredtext'

import numpy as np


def output_T(results, contrast, retvals=('effect', 'sd', 't')):
    """ Convenience function to collect t contrast results

    Parameters
    ----------
    results : object
        implementing Tcontrast method
    contrast : array
        contrast matrix
    retvals : sequence, optional
        None or more of strings 'effect', 'sd', 't', where the presence of the
        string means that that output will be returned.

    Returns
    -------
    res_list : list
        List of results.  It will have the same length as `retvals` and the
        elements will be in the same order as retvals
    """
    r = results.Tcontrast(contrast, store=retvals)
    returns = []
    for valname in retvals:
        if valname == 'effect':
            returns.append(r.effect)
        if valname == 'sd':
            returns.append(r.sd)
        if valname == 't':
            returns.append(r.t)
    return returns


def output_F(results, contrast):
    """
    This convenience function outputs the results of an Fcontrast
    from a regression

    Parameters
    ----------
    results : object
        implementing Tcontrast method
    contrast : array
        contrast matrix

    Returns
    -------
    F : array
        array of F values
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
    Output contrast related to a T contrast from a GLM pass through data.
    """
    def __init__(self, contrast, effect=None, sd=None, t=None):
        # Returns a list of arrays, being [effect, sd, t] when all these are not
        # None
        # Compile list of desired return values
        retvals = []
        # Set self.list to contain selected input catching objects
        self.list = []
        if not effect is None:
            retvals.append('effect')
            self.list.append(effect)
        if not sd is None:
            retvals.append('sd')
            self.list.append(sd)
        if not t is None:
            retvals.append('t')
            self.list.append(t)
        # Set return function to return selected inputs
        self.fn = lambda x: output_T(x, contrast, retvals)


def output_AR1(results):
    """
    Compute the usual AR(1) parameter on
    the residuals from a regression.
    """
    resid = results.resid
    rho = np.add.reduce(resid[0:-1]*resid[1:] / np.add.reduce(resid[1:-1]**2))
    return rho


