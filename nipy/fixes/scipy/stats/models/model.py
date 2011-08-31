# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from numpy.linalg import inv
#from scipy import optimize

from scipy.stats import t

from nipy.algorithms.utils.matrices import pos_recipr
import numpy.lib.recfunctions as nprf
from descriptors import setattr_on_read

class Model(object):
    """
    A (predictive) statistical model. The class Model itself does nothing
    but lays out the methods expected of any subclass.
    """

    def __init__(self):
        pass

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance. For
        instance, the design matrix of a linear model may change
        and some things must be recomputed.
        """
        raise NotImplementedError

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

    def predict(self, design=None):
        """
        After a model has been fit, results are (assumed to be) stored
        in self.results, which itself should have a predict method.
        """
        self.results.predict(design)

class LikelihoodModel(Model):

    def logL(self, theta, Y, nuisance=None):
        """
        Log-likelihood of model.
        """
        raise NotImplementedError

    def score(self, theta, Y, nuisance=None):
        """
        Score function of model = gradient of logL with respect to
        theta.
        """
        raise NotImplementedError

    def information(self, theta, nuisance=None):
        """
        Fisher information matrix: the inverse of the expected value of -d^2 logL/dtheta^2.
        """
        raise NotImplementedError


class LikelihoodModelResults(object):
# Jonathan: This is the class in which
# things like AIC, BIC, llf would be implemented as methods,
# not computed in, say, the fit method of OLSModel

    ''' Class to contain results from likelihood models '''
    def __init__(self, theta, Y, model, cov=None, dispersion=1., nuisance=None,
                 rank=None):
        ''' Set up results structure

        Parameters
        ----------
        theta : ndarray
            parameter estimates from estimated model
        Y : ndarray
            data
        model : ``LikelihoodModel`` instance
            model used to generate fit
        cov : None or ndarray, optional
            covariance of thetas
        dispersion : scalar, optional
            multiplicative factor in front of `cov`
        nuisance : None of ndarray
            parameter estimates needed to compute logL
        rank : None or scalar
            rank of the model.  If rank is not None, it is used for df_model
            instead of the usual counting of parameters.

        Notes
        -----
        The covariance of thetas is given by:

            dispersion * cov

        For (some subset of models) `dispersion` will typically be the mean
        square error from the estimated model (sigma^2)
        '''
        self.theta = theta
        self.Y = Y
        self.model = model
        if cov is None:
            self.cov = self.model.information(self.theta, nuisance=self.nuisance)
        else:
            self.cov = cov
        self.dispersion = dispersion
        self.nuisance = nuisance

        self.df_total = Y.shape[0]
        self.df_model = model.df_model # put this as a parameter of LikelihoodModel
        self.df_resid = self.df_total - self.df_model

    @setattr_on_read
    def logL(self):
        """
        The maximized log-likelihood
        """
        return self.model.logL(self.theta, self.Y, nuisance=self.nuisance)

    @setattr_on_read
    def AIC(self):
        """
        Akaike Information Criterion
        """
        p = self.theta.shape[0]
        return -2*self.logL + 2*p

    @setattr_on_read
    def BIC(self):
        """
        Schwarz's Bayesian Information Criterion
        """
        n = self.Y.shape[0]
        p = self.theta.shape[0]
        return -2*self.logL + np.log(n)*p

    def t(self, column=None):
        """
        Return the (Wald) t-statistic for a given parameter estimate.

        Use Tcontrast for more complicated (Wald) t-statistics.

        """

        if column is None:
            column = range(self.theta.shape[0])

        column = np.asarray(column)
        _theta = self.theta[column]
        _cov = self.vcov(column=column)
        if _cov.ndim == 2:
            _cov = np.diag(_cov)
        _t = _theta * pos_recipr(np.sqrt(_cov))
        return _t

    def vcov(self, matrix=None, column=None, dispersion=None, other=None):
        """ Variance/covariance matrix of linear contrast

        Returns the variance/covariance matrix of a linear contrast of the
        estimates of theta, multiplied by `dispersion` which will often be an
        estimate of `dispersion`, like, sigma^2.

        The covariance of interest is either specified as a (set of) column(s)
        or a matrix.
        """
        if self.cov is None:
            raise ValueError, 'need covariance of parameters for computing (unnormalized) covariances'

        if dispersion is None:
            dispersion = self.dispersion

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return self.cov[column, column] * dispersion
            else:
                return self.cov[column][:,column] * dispersion

        elif matrix is not None:
            if other is None:
                other = matrix
            tmp = np.dot(matrix, np.dot(self.cov, np.transpose(other)))
            return tmp * dispersion

        if matrix is None and column is None:
            return self.cov * dispersion

# need to generalize for the case when dispersion is not a scalar
# and we have robust estimates of \Omega the error covariance matrix
#
# Jonathan: here, dispersion's shape has NOTHING to do with 
# resid.shape. In the nipy applications, the same model is fit
# at 1000s of voxels at once, each one having a separate dispersion estimate.
# The fix below seems to assume heteroscedastic errors.
# this is what the class WLSModel is for...
#
#             if dispersion.size==1:
#                 dispersion=np.eye(len(self.resid))*dispersion
#             return np.dot(np.dot(self.calc_theta, dispersion), self.calc_theta.T)

    def Tcontrast(self, matrix, t=True, sd=True, dispersion=None):
        """
        Compute a Tcontrast for a row vector matrix. To get the t-statistic
        for a single column, use the 't' method.
        """

        _t = _sd = None

        _effect = np.dot(matrix, self.theta)

        if sd:
            _sd = np.sqrt(self.vcov(matrix=matrix, dispersion=dispersion))
        if t:
            _t = _effect * pos_recipr(_sd)
        return TContrastResults(effect=_effect, t=_t, sd=_sd, df_den=self.df_resid)

# Jonathan: for an F-statistic, the options 't', 'sd' do not make sense. The 'effect' option
# does make sense, but is rarely looked at in practice.
# Usually, you just want the F-statistic.
#    def Fcontrast(self, matrix, eff=True, t=True, sd=True, dispersion=None, invcov=None):

    def Fcontrast(self, matrix, dispersion=None, invcov=None):
        """
        Compute an Fcontrast for a contrast matrix.

        Here, matrix M is assumed to be non-singular. More precisely,

        M pX pX' M'

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.

        See the contrast module to see how to specify contrasts.
        In particular, the matrices from these contrasts will always be
        non-singular in the sense above.
        """

        ctheta = np.dot(matrix, self.theta)

        if matrix.ndim == 1:
            matrix = matrix.reshape((1, matrix.shape[0]))

        if dispersion is None:
            dispersion = self.dispersion

        q = matrix.shape[0]
        if invcov is None:
            invcov = inv(self.vcov(matrix=matrix, dispersion=1.0))
        F = np.add.reduce(np.dot(invcov, ctheta) * ctheta, 0) * pos_recipr((q * dispersion))
        return FContrastResults(F=F, df_den=self.df_resid, df_num=invcov.shape[0])

    def conf_int(self, alpha=.05, cols=None, dispersion=None):
        '''
        Returns the confidence interval of the specified theta estimates.

        Parameters
        ----------
        alpha : float, optional
            The `alpha` level for the confidence interval.
            ie., `alpha` = .05 returns a 95% confidence interval.
        cols : tuple, optional
            `cols` specifies which confidence intervals to return
        dispersion : None or scalar
            scale factor for the variance / covariance (see class docstring and
            ``vcov`` method docstring)

        Returns
        -------
        cis : ndarray
            `cis` is shape ``(len(cols), 2)`` where each row contains [lower,
            upper] for the given entry in `cols`

        Example
        -------
        >>> from numpy.random import standard_normal as stan
        >>> from nipy.fixes.scipy.stats.models.regression import OLSModel
        >>> x = np.hstack((stan((30,1)),stan((30,1)),stan((30,1))))
        >>> beta=np.array([3.25, 1.5, 7.0])
        >>> y = np.dot(x,beta) + stan((30))
        >>> model = OLSModel(x, hascons=False).fit(y)
        >>> confidence_intervals = model.conf_int(cols=(1,2))

        Notes
        -----
        Confidence intervals are two-tailed.
        TODO:
        tails : string, optional
            `tails` can be "two", "upper", or "lower"
        '''
        if cols is None:
            lower = self.theta - t.ppf(1-alpha/2,self.df_resid) *\
                    np.diag(np.sqrt(self.vcov(dispersion=dispersion)))
            upper = self.theta + t.ppf(1-alpha/2,self.df_resid) *\
                    np.diag(np.sqrt(self.vcov(dispersion=dispersion)))
        else:
            lower=[]
            upper=[]
            for i in cols:
                lower.append(self.theta[i] - t.ppf(1-alpha/2,self.df_resid) *\
                    np.diag(np.sqrt(self.vcov(dispersion=dispersion)))[i])
                upper.append(self.theta[i] + t.ppf(1-alpha/2,self.df_resid) *\
                    np.diag(np.sqrt(self.vcov(dispersion=dispersion)))[i])
        return np.asarray(zip(lower,upper))


class TContrastResults(object):
    """
    Results from looking at a particular contrast of coefficients in
    a parametric model. The class does nothing, it is a container
    for the results from T contrasts, and returns the T-statistics
    when np.asarray is called.
    """

    def __init__(self, t, sd, effect, df_den=None):
        if df_den is None:
            df_den = np.inf
        self.t = t
        self.sd = sd
        self.effect = effect
        self.df_den = df_den

    def __array__(self):
        return np.asarray(self.t)
        
    def __str__(self):
        return '<T contrast: effect=%s, sd=%s, t=%s, df_den=%d>' % \
            (`self.effect`, `self.sd`, `self.t`, self.df_den)



class FContrastResults(object):
    """
    Results from looking at a particular contrast of coefficients in
    a parametric model. The class does nothing, it is a container
    for the results from F contrasts, and returns the F-statistics
    when np.asarray is called.
    """

    def __init__(self, F, df_num, df_den=None):
        if df_den is None:
            df_den = np.inf
        self.F = F
        self.df_den = df_den
        self.df_num = df_num

    def __array__(self):
        return np.asarray(self.F)
        
    def __str__(self):
        return '<F contrast: F=%s, df_den=%d, df_num=%d>' % \
            (`self.F`, self.df_den, self.df_num)

