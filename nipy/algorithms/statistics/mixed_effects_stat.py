"""
Module for computation of mixed effects statistics with an EM algorithm.
i.e.
solves problems of the form
y = X beta + e1 + e2,
where X and Y are known, e1 and e2 are centered with diagonal covariance.
V1 = var(e1) is known, and V2 = var(e2) = lambda identity.
the code estimates beta and lambda using an EM algorithm.
Likelihood ratio tests can then be used to test the columns of beta.

Author: Bertrand Thirion, 2012.

>>> N, P = 15, 500
>>> V1 = np.random.randn(N, P) ** 2
>>> effects = np.ones(P)
>>> Y = generate_data(np.ones(N), effects, .25, V1)
>>> T1 = one_sample_ttest(Y, V1, n_iter=5)
>>> T2 = t_stat(Y)
>>> assert(T1.std() < T2.std())
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

EPS = 100 * np.finfo(float).eps


def generate_data(X, beta, V2, V1):
    """ Generate a group of individuals from the provided parameters

    Parameters
    ----------
    X: array of shape (n_samples, n_reg),
       the design matrix of the model
    beta: float or array of shape (n_reg, n_tests),
          the associated effects
    V2: float or array of shape (n_tests),
              group variance
    V1: array of shape(n_samples, n_tests),
               the individual variances

    Returns
    -------
    Y: array of shape(n_samples, n_tests)
       the individual data related to the two-level normal model
    """
    # check that the variances are  positive
    if (V1 < 0).any():
        raise ValueError('Variance should be positive')
    Y = np.random.randn(*V1.shape)
    Y *= np.sqrt(V2 + V1)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if np.isscalar(beta):
        beta = beta * np.ones((X.shape[1], V1.shape[1]))
    if beta.ndim == 1:
        beta = beta[np.newaxis]

    Y += np.dot(X, beta)
    return Y


def check_arrays(Y, V1):
    """Check that the given data can be used for the models

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests) or (n_samples)
       the estimated effects
    V1: array of shape (n_samples, n_tests) or (n_samples)
          first-level variance
    """
    if (V1 < 0).any():
        raise ValueError("a negative variance has been provided")

    if np.size(Y) == Y.shape[0]:
        Y = Y[:, np.newaxis]

    if np.size(V1) == V1.shape[0]:
        V1 = V1[:, np.newaxis]

    if Y.shape != V1.shape:
        raise ValueError("Y and V1 do not have the same shape")
    return Y, V1


def t_stat(Y):
    """ Returns the t stat of the sample on each row of the matrix

    Parameters
    ----------
    Y, array of shape (n_samples, n_tests)

    Returns
    -------
    t_variates, array of shape (n_tests)
    """
    return Y.mean(0) / Y.std(0) * np.sqrt(Y.shape[0] - 1)


class MixedEffectsModel(object):
    """Class to handle multiple one-sample mixed effects models
    """

    def __init__(self, X, n_iter=5, verbose=False):
        """
        Set the effects and first-level variance,
        and initialize related quantities

        Parameters
        ----------
        X: array of shape(n_samples, n_effects),
           the design matrix
        n_iter: int, optional,
               number of iterations of the EM algorithm
        verbose: bool, optional, verbosity mode
        """
        self.n_iter = n_iter
        self.verbose = verbose
        self.X = X
        self.pinv_X = np.linalg.pinv(X)

    def log_like(self, Y, V1):
        """ Compute the log-likelihood of (Y, V1) under the model

        Parameters
        ----------
        Y, array of shape (n_samples, n_tests) or (n_samples)
           the estimated effects
        V1, array of shape (n_samples, n_tests) or (n_samples)
            first-level variance

        Returns
        -------
        logl: array of shape self.n_tests,
              the log-likelihood of the model
        """
        Y, V1 = check_arrays(Y, V1)
        tvar = self.V2 + V1
        logl = np.sum(((Y - self.Y_) ** 2) / tvar, 0)
        logl += np.sum(np.log(tvar), 0)
        logl += np.log(2 * np.pi) * Y.shape[0]
        logl *= (- 0.5)
        return logl

    def predict(self, Y, V1):
        """Return the log_likelihood of the data.See the log_like method"""
        return self.log_like(Y, V1)

    def score(self, Y, V1):
        """Return the log_likelihood of the data. See the log_like method"""
        return self.log_like(Y, V1)

    def _one_step(self, Y, V1):
        """Applies one step of an EM algorithm to estimate self.mean_, self.var

        Parameters
        ----------
        Y, array of shape (n_samples, n_tests) or (n_samples)
              the estimated effects
        V1, array of shape (n_samples, n_tests) or (n_samples)
                 first-level variance
        """
        # E step
        prec = 1. / (self.V2 + V1)
        Y_ = prec * (self.V2 * Y + V1 * self.Y_)
        cvar = V1 * self.V2 * prec

        # M step
        self.beta_ = np.dot(self.pinv_X, Y_)
        self.Y_ = np.dot(self.X, self.beta_)
        self.V2 = np.mean((Y_ - self.Y_) ** 2, 0) + cvar.mean(0)

    def fit(self, Y, V1):
        """ Launches the EM algorithm to estimate self

        Parameters
        ----------
        Y, array of shape (n_samples, n_tests) or (n_samples)
              the estimated effects
        V1, array of shape (n_samples, n_tests) or (n_samples)
                 first-level variance

        Returns
        -------
        self
        """
        # Basic data checks
        if self.X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same numbers of rows')
        Y, V1 = check_arrays(Y, V1)
        self.beta_ = np.dot(self.pinv_X, Y)
        self.Y_ = np.dot(self.X, self.beta_)
        self.V2 = np.mean((Y - self.Y_) ** 2, 0)

        if self.verbose:
            log_like_init = self.log_like(Y, V1)
            print('Average log-likelihood: ', log_like_init.mean())

        for i in range(self.n_iter):
            self._one_step(Y, V1)

            if self.verbose:
                log_like_ = self.log_like(Y, V1)
                if (log_like_ < (log_like_init - EPS)).any():
                    raise ValueError('The log-likelihood cannot decrease')
                log_like_init = log_like_
                print('Iteration %d, average log-likelihood: %f' % (
                        i, log_like_.mean()))
        return self


def two_sample_ftest(Y, V1, group, n_iter=5, verbose=False):
    """Returns the mixed effects t-stat for each row of the X
    (one sample test)
    This uses the Formula in Roche et al., NeuroImage 2007

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the data
    V1: array of shape (n_samples, n_tests)
        first-level variance assocated with the data
    group: array of shape (n_samples)
       a vector of indicators yielding the samples membership
    n_iter: int, optional,
           number of iterations of the EM algorithm
    verbose: bool, optional, verbosity mode

    Returns
    -------
    tstat: array of shape (n_tests),
           statistical values obtained from the likelihood ratio test
    """
    # check that group is correct
    if group.size != Y.shape[0]:
        raise ValueError('The number of labels is not the number of samples')
    if (np.unique(group) != np.array([0, 1])).all():
        raise ValueError('group should be composed only of zeros and ones')

    # create design matrices
    X = np.vstack((np.ones_like(group), group)).T
    return mfx_stat(Y, V1, X, 1, n_iter=n_iter, verbose=verbose,
                    return_t=False, return_f=True)[0]


def two_sample_ttest(Y, V1, group, n_iter=5, verbose=False):
    """Returns the mixed effects t-stat for each row of the X
    (one sample test)
    This uses the Formula in Roche et al., NeuroImage 2007

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the data
    V1: array of shape (n_samples, n_tests)
        first-level variance assocated with the data
    group: array of shape (n_samples)
       a vector of indicators yielding the samples membership
    n_iter: int, optional,
           number of iterations of the EM algorithm
    verbose: bool, optional, verbosity mode

    Returns
    -------
    tstat: array of shape (n_tests),
           statistical values obtained from the likelihood ratio test
    """
    X = np.vstack((np.ones_like(group), group)).T
    return mfx_stat(Y, V1, X, 1, n_iter=n_iter, verbose=verbose,
                    return_t=True)[0]


def one_sample_ftest(Y, V1, n_iter=5, verbose=False):
    """Returns the mixed effects F-stat for each row of the X
    (one sample test)
    This uses the Formula in Roche et al., NeuroImage 2007

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the data
    V1: array of shape (n_samples, n_tests)
        first-level variance ssociated with the data
    n_iter: int, optional,
           number of iterations of the EM algorithm
    verbose: bool, optional, verbosity mode

    Returns
    -------
    fstat, array of shape (n_tests),
           statistical values obtained from the likelihood ratio test
    sign, array of shape (n_tests),
          sign of the mean for each test (allow for post-hoc signed tests)
    """
    return mfx_stat(Y, V1, np.ones((Y.shape[0], 1)), 0, n_iter=n_iter,
                    verbose=verbose, return_t=False, return_f=True)[0]


def one_sample_ttest(Y, V1, n_iter=5, verbose=False):
    """Returns the mixed effects t-stat for each row of the X
    (one sample test)
    This uses the Formula in Roche et al., NeuroImage 2007

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the observations
    V1: array of shape (n_samples, n_tests)
        first-level variance associated with the observations
    n_iter: int, optional,
           number of iterations of the EM algorithm
    verbose: bool, optional, verbosity mode

    Returns
    -------
    tstat: array of shape (n_tests),
           statistical values obtained from the likelihood ratio test
    """
    return mfx_stat(Y, V1, np.ones((Y.shape[0], 1)), 0, n_iter=n_iter,
                    verbose=verbose, return_t=True)[0]


def mfx_stat(Y, V1, X, column, n_iter=5, return_t=True,
             return_f=False, return_effect=False,
             return_var=False, verbose=False):
    """Run a mixed-effects model test on the column of the design matrix

    Parameters
    ----------
    Y: array of shape (n_samples, n_tests)
       the data
    V1: array of shape (n_samples, n_tests)
        first-level variance assocated with the data
    X: array of shape(n_samples, n_regressors)
       the design matrix of the model
    column: int,
            index of the column of X to be tested
    n_iter: int, optional,
           number of iterations of the EM algorithm
    return_t: bool, optional,
              should one return the t test (True by default)
    return_f: bool, optional,
              should one return the F test (False by default)
    return_effect: bool, optional,
              should one return the effect estimate (False by default)
    return_var: bool, optional,
              should one return the variance estimate (False by default)

    verbose: bool, optional, verbosity mode

    Returns
    -------
    (tstat, fstat, effect, var): tuple of arrays of shape (n_tests),
                                 those required by the input return booleans

    """
    # check that X/columns are correct
    column = int(column)
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X.shape[0] is not the number of samples')
    if (column > X.shape[1]):
        raise ValueError('the column index is more than the number of columns')

    # create design matrices
    contrast_mask = 1 - np.eye(X.shape[1])[column]
    X0 = X * contrast_mask

    # instantiate the mixed effects models
    model_0 = MixedEffectsModel(X0, n_iter=n_iter, verbose=verbose).fit(Y, V1)
    model_1 = MixedEffectsModel(X, n_iter=n_iter, verbose=verbose).fit(Y, V1)

    # compute the log-likelihood ratio statistic
    fstat = 2 * (model_1.log_like(Y, V1) - model_0.log_like(Y, V1))
    fstat = np.maximum(0, fstat)
    sign = np.sign(model_1.beta_[column])

    output = ()
    if return_t:
        output += (np.sqrt(fstat) * sign,)
    if return_f:
        output += (fstat,)
    if return_var:
        output += (model_1.V2,)
    if return_effect:
        output += (model_1.beta_[column],)
    return output
