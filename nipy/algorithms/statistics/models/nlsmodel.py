# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Non-linear least squares model
"""
from __future__ import absolute_import
__docformat__ = 'restructuredtext'

import numpy as np
import numpy.linalg as npl

from .model import Model

from nipy.externals.six import Iterator


class NLSModel(Model, Iterator):
    """
    Class representing a simple nonlinear least squares model.
    """

    def __init__(self, Y, design, f, grad, theta, niter=10):
        """ Initialize non-linear model instance

        Parameters
        ----------
        Y : ndarray
            the data in the NLS model
        design : ndarray
            the design matrix, X
        f : callable
            the map between the (linear parameters (in the design matrix) and
            the nonlinear parameters (theta)) and the predicted data. `f`
            accepts the design matrix and the parameters (theta) as input, and
            returns the predicted data at that design.
        grad : callable
            the gradient of f, this should be a function of an nxp design
            matrix X and qx1 vector theta that returns an nxq matrix
            df_i/dtheta_j where:

            .. math::

                f_i(theta) = f(X[i], theta)

            is the nonlinear response function for the i-th instance in
            the model.
        theta : array
            parameters
        niter : int
            number of iterations
        """
        Model.__init__(self)
        self.Y = Y
        self.design = design
        self.f = f
        self.grad = grad
        self.theta = theta
        self.niter = niter
        if self.design is not None and self.Y is not None:
            if self.Y.shape[0] != self.design.shape[0]:
                raise ValueError('Y should be same shape as design')

    def _Y_changed(self):
        if self.design is not None:
            if self.Y.shape[0] != self.design.shape[0]:
                raise ValueError('Y should be same shape as design')

    def _design_changed(self):
        if self.Y is not None:
            if self.Y.shape[0] != self.design.shape[0]:
                raise ValueError('Y should be same shape as design')

    def getZ(self):
        """ Set Z into `self`

        Returns
        -------
        None
        """
        self._Z = self.grad(self.design, self.theta)

    def getomega(self):
        """ Set omega into `self`

        Returns
        -------
        None
        """
        self._omega = self.predict() - np.dot(self._Z, self.theta)

    def predict(self, design=None):
        """ Get predicted values for `design` or ``self.design``

        Parameters
        ----------
        design : None or array, optional
            design at which to predict data.  If None (the default) then use the
            initial ``self.design``

        Returns
        -------
        y_predicted : array
            predicted data at given (or initial) design
        """
        if design is None:
            design = self.design
        return self.f(design, self.theta)

    def SSE(self):
        """ Sum of squares error.

        Returns
        -------
        sse: float
            sum of squared residuals
        """
        return sum((self.Y - self.predict()) ** 2)

    def __iter__(self):
        """ Get iterator from model instance

        Returns
        -------
        itor : iterator
            Returns ``self``
        """
        if self.theta is not None:
            self.initial = self.theta
        elif self.initial is not None:
            self.theta = self.initial
        else:
            raise ValueError('need an initial estimate for theta')

        self._iter = 0
        self.theta = self.initial
        return self

    def __next__(self):
        """ Do an iteration of fit

        Returns
        -------
        None
        """
        if self._iter < self.niter:
            self.getZ()
            self.getomega()
            Zpinv = npl.pinv(self._Z)
            self.theta = np.dot(Zpinv, self.Y - self._omega)
        else:
            raise StopIteration
        self._iter += 1
