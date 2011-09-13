# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
__docformat__ = 'restructuredtext'

import numpy as np

class VarianceFunction(object):
    """
    Variance function that relates the variance of a random variable
    to its mean. Defaults to 1.
    """

    def __call__(self, mu):
        """
        Default variance function

        INPUTS:
           mu  -- mean parameters

        OUTPUTS: v
           v   -- ones(mu.shape)
        """

        return np.ones(mu.shape, np.float64)

constant = VarianceFunction()

class Power(object):
    """
    Power variance function:

    V(mu) = fabs(mu)**power

    INPUTS:
       power -- exponent used in power variance function

    """

    def __init__(self, power=1.):
        self.power = power

    def __call__(self, mu):

        """
        Power variance function

        INPUTS:
           mu  -- mean parameters

        OUTPUTS: v
           v   -- fabs(mu)**self.power
        """
        return np.power(np.fabs(mu), self.power)

class Binomial(object):
    """
    Binomial variance function

    p = mu / n; V(mu) = p * (1 - p) * n

    INPUTS:
       n -- number of trials in Binomial
    """

    tol = 1.0e-10

    def __init__(self, n=1):
        self.n = n

    def clean(self, p):
        return np.clip(p, Binomial.tol, 1 - Binomial.tol)

    def __call__(self, mu):
        """
        Binomial variance function

        INPUTS:
           mu  -- mean parameters

        OUTPUTS: v
           v   -- mu / self.n * (1 - mu / self.n) * self.n
        """
        p = self.clean(mu / self.n)
        return p * (1 - p) * self.n

mu = Power()
mu_squared = Power(power=2)
mu_cubed = Power(power=3)
binary = Binomial()
