# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module implements some standard regression models: OLS and WLS
models, as well as an AR(p) regression model.

Models are specified with a design matrix and are fit using their
'fit' method.

Subclasses that have more complicated covariance matrices
should write over the 'whiten' method as the fit method
prewhitens the response by calling 'whiten'.

General reference for regression models:

'Introduction to Linear Regression Analysis', Douglas C. Montgomery,
    Elizabeth A. Peck, G. Geoffrey Vining. Wiley, 2006.

"""

__docformat__ = 'restructuredtext en'

import warnings

import numpy as np
import numpy.lib.recfunctions as nprf

from scipy import stats
import scipy.linalg as spl

from .model import LikelihoodModel, LikelihoodModelResults
from . import utils

from .descriptors import setattr_on_read

def categorical(data):
    '''
    Returns an array changing categorical variables to dummy variables.

    Take a structured or record array and returns an array with categorical variables.

    Notes
    -----
    This returns a dummy variable for EVERY distinct string.  If noconsant
    then this is okay.  Otherwise, a "intercept" needs to be designated in regression.

    Returns the same array as it's given right now, consider returning a structured 
    and plain ndarray (with names stripped, etc.)
    '''
    if not data.dtype.names and not data.mask.any():
        print data.dtype
        print "There is not a categorical variable?"
        return data
    #if data.mask.any():
    #    print "Masked arrays are not handled yet."
    #    return data

    elif data.dtype.names:  # this will catch both structured and record
                            # arrays, no other array could have string data!
                            # not sure about masked arrays yet
        for i in range(len(data.dtype)):
            if data.dtype[i].type is np.string_:
                tmp_arr = np.unique(data.field(i))
                tmp_dummy = (tmp_arr[:,np.newaxis]==data.field(i)).astype(float)
# .field only works for record arrays
# tmp_dummy is a number of dummies x number of observations array
                data=nprf.drop_fields(data,data.dtype.names[i],usemask=False, 
                                asrecarray=True)
                data=nprf.append_fields(data,tmp_arr.strip("\""), data=tmp_dummy,
                                    usemask=False, asrecarray=True)
        return data


#How to document a class?
#Docs are a little vague and there are no good examples
#Some of these attributes are most likely intended to be private I imagine
class OLSModel(LikelihoodModel):
    """    
    A simple ordinary least squares model.

    Parameters
    ----------
        `design`: array-like
            This is your design matrix.  Data are assumed to be column ordered
            with observations in rows.

    Methods
    -------
    model.logL(b=self.beta, Y)
        Returns the log-likelihood of the parameter estimates

        Parameters
        ----------
        b : array-like
            `b` is an array of parameter estimates the log-likelihood of which 
            is to be tested.
        Y : array-like
            `Y` is the vector of dependent variables.            
    model.__init___(design, hascons=True)
        Creates a `OLSModel` from a design.

    Attributes
    ----------
    design : ndarray
        This is the design, or X, matrix.
    wdesign : ndarray
        This is the whitened design matrix.
        design = wdesign by default for the OLSModel, though models that
        inherit from the OLSModel will whiten the design.
    calc_beta : ndarray
        This is the Moore-Penrose pseudoinverse of the whitened design matrix.
    normalized_cov_beta : ndarray
        np.dot(calc_beta, calc_beta.T)
    df_resid : integer
        Degrees of freedom of the residuals.
        Number of observations less the rank of the design.
    df_model : integer
        Degres of freedome of the model.
        The rank of the design.

    Examples
    --------
    >>> from nipy.modalities.fmri.formula import Term, Formula
    >>> data = np.rec.fromarrays(([1,3,4,5,2,3,4], range(1,8)), names=('Y', 'X'))
    >>> f = Formula([Term("X"), 1])
    >>> dmtx = f.design(data, return_float=True)
    >>> model = OLSModel(dmtx)
    >>> results = model.fit(data['Y'])
    >>> results.theta
    array([ 0.25      ,  2.14285714])
    >>> results.t()
    array([ 0.98019606,  1.87867287])
    >>> print results.Tcontrast([0,1]) #doctest: +FP_6DP
    <T contrast: effect=2.14285714286, sd=1.14062281591, t=1.87867287326, df_den=5>
    >>> print results.Fcontrast(np.eye(2)) #doctest: +FP_6DP
    <F contrast: F=19.4607843137, df_den=5, df_num=2>
    """
    def __init__(self, design, hascons=True):
        super(OLSModel, self).__init__()
        self.initialize(design, hascons)

    def initialize(self, design, hascons=True):
# Jonathan: PLEASE don't assume we have a constant...
# TODO: handle case for noconstant regression
        self.design = design
        self.wdesign = self.whiten(self.design)
        self.calc_beta = spl.pinv(self.wdesign)
        self.normalized_cov_beta = np.dot(self.calc_beta,
                                          np.transpose(self.calc_beta))
        self.df_total = self.wdesign.shape[0]
        self.df_model = utils.rank(self.design)
        self.df_resid = self.df_total - self.df_model

    def logL(self, beta, Y, nuisance=None):
        # Jonathan: this is overwriting an abstract method of LikelihoodModel
        '''
        Returns the value of the loglikelihood function at beta.
        
        Given the whitened design matrix, the loglikelihood is evaluated
        at the parameter vector, beta, for the dependent variable, Y
        and the nuisance parameter, sigma.

        Parameters
        ----------

        beta : ndarray
            The parameter estimates.  Must be of length df_model.

        Y : ndarray
            The dependent variable.

        nuisance : dict, optional
            A dict with key 'sigma', which is an optional 
            estimate of sigma. If None, defaults to its
            maximum likelihood estimate (with beta fixed)
            as
            
            sum((Y - X*beta)**2) / n

            where n=Y.shape[0], X=self.design.

        Returns
        -------
        The value of the loglikelihood function.
        

        Notes
        -----
        The log-Likelihood Function is defined as
        .. math:: \ell(\beta,\sigma,Y)=
        -\frac{n}{2}\log(2\pi\sigma^2) - \|Y-X\beta\|^2/(2\sigma^2)
        ..

        The parameter :math:`\sigma` above is what is sometimes
        referred to as a nuisance parameter. That is, the likelihood
        is considered as a function of :math:`\beta`, but to evaluate it,
        a value of :math:`\sigma` is needed.

        If :math:`\sigma` is not provided, then its maximum likelihood
        estimate
        .. math::\hat{\sigma}(\beta) = \frac{\text{SSE}(\beta)}{n}

        is plugged in. This likelihood is now a function
        of only :math:`\beta` and is technically referred to as 
        a profile-likelihood.

        References
        ----------
        .. [1] W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.
        '''
        X = self.wdesign
        wY = self.whiten(Y)
        r = wY - np.dot(X, beta)
        n = self.df_total
        SSE = (r**2).sum(0)
        if nuisance is None:
            sigmasq = SSE / n
        else:
            sigmasq = nuisance['sigma']
        loglf = -n/2.*np.log(2*np.pi*sigmasq) - SSE / (2*sigmasq)
        return loglf
    
    def score(self, beta, Y, nuisance=None):
        # Jonathan: this is overwriting an abstract method of LikelihoodModel
        '''
        Returns the score function, the gradient of the loglikelihood function at (beta, Y, nuisance).

        See logL for details.
        
        Parameters
        ----------

        beta : ndarray
            The parameter estimates.  Must be of length df_model.

        Y : ndarray
            The dependent variable.

        nuisance : dict, optional
            A dict with key 'sigma', which is an optional 
            estimate of sigma. If None, defaults to its
            maximum likelihood estimate (with beta fixed)
            as
            
            sum((Y - X*beta)**2) / n

            where n=Y.shape[0], X=self.design.

        Returns
        -------
        The gradient of the loglikelihood function.
        '''
        X = self.wdesign
        wY = self.whiten(Y)
        r = wY - np.dot(X, beta)
        n = self.df_total
        if nuisance is None:
            SSE = (r**2).sum(0)
            sigmasq = SSE / n
        else:
            sigmasq = nuisance['sigma']
        return np.dot(X, r) / sigmasq

    def information(self, beta, nuisance=None):
        # Jonathan: this is overwriting an abstract method of LikelihoodModel
        '''
        Returns the information matrix at (beta, Y, nuisance).
        
        See logL for details.

        Parameters
        ----------

        beta : ndarray
            The parameter estimates.  Must be of length df_model.

        nuisance : dict
            A dict with key 'sigma', which is an 
            estimate of sigma. If None, defaults to its
            maximum likelihood estimate (with beta fixed)
            as
            
            sum((Y - X*beta)**2) / n

            where n=Y.shape[0], X=self.design.

        Returns
        -------
        The information matrix, the negative of the inverse of the
        Hessian of the
        of the log-likelihood function evaluated at (theta, Y, nuisance).
        

        '''
        X = self.design
        sigmasq = nuisance['sigma']
        C = sigmasq * np.dot(X.T, X)
        return C

                      
#   Note: why have a function that doesn't do anything? does it have to be here to be
#   overwritten?
#   Could this be replaced with the sandwich estimators 
#   without writing a subclass?
#
#   Jonathan: the subclasses WLSModel, ARModel and GLSModel all 
#   overwrite this method. The point of these subclasses
#   is such that not much of OLSModel has to be changed

    def whiten(self, X):
        """
        This matrix is the matrix whose pseudoinverse is ultimately
        used in estimating the coefficients. For OLSModel, it is
        does nothing. For WLSmodel, ARmodel, it pre-applies
        a square root of the covariance matrix to X.
        """
        return X

    @setattr_on_read
    def has_intercept(self):
        """
        Check if column of 1s is in column space of design
        """
        o = np.ones(self.design.shape[0])
        obeta = np.dot(self.calc_beta, o)
        ohat = np.dot(self.wdesign, obeta)
        if np.allclose(ohat, o):
            return True
        return False

    @setattr_on_read
    def rank(self):
        """
        Compute rank of design matrix
        """
        return utils.rank(self.wdesign)

    def fit(self, Y):
#    def fit(self, Y, robust=None):
# Jonathan: it seems the robust method are different estimates
# of the covariance matrix for a heteroscedastic regression model.
# This functionality is in WLSmodel. (Weighted least squares models assume
# covariance is diagonal, i.e. heteroscedastic).

# Some of the quantities, like AIC and BIC are defined for 
# any model with a likelihood and they should be properties
# of the LikelihoodModel
        """
        Full fit of the model including estimate of covariance matrix,
        (whitened) residuals and scale.

        Parameters
        ----------
        Y : array-like
            The dependent variable for the Least Squares problem.

        Returns
        -------
        fit : RegressionResults
        """
        wY = self.whiten(Y)
        beta = np.dot(self.calc_beta, wY)
        wresid = wY - np.dot(self.wdesign, beta)
        dispersion = np.sum(wresid**2, 0) / (self.wdesign.shape[0] - self.wdesign.shape[1])
        lfit = RegressionResults(beta, Y, self,
                                 wY, wresid, dispersion=dispersion,
                                 cov=self.normalized_cov_beta)
        return lfit


class ARModel(OLSModel):
    """
    A regression model with an AR(p) covariance structure.

    In terms of a LikelihoodModel, the parameters
    are beta, the usual regression parameters,
    and sigma, a scalar nuisance parameter that
    shows up as multiplier in front of the AR(p) covariance.

    The linear autoregressive process of order p--AR(p)--is defined as:
        TODO

    Examples
    --------
    >>> from nipy.modalities.fmri.formula import Term, Formula
    >>> data = np.rec.fromarrays(([1,3,4,5,8,10,9], range(1,8)), names=('Y', 'X'))
    >>> f = Formula([Term("X"), 1])
    >>> dmtx = f.design(data, return_float=True)
    >>> model = ARModel(dmtx, 2)

    We go through the ``model.iterative_fit`` procedure long-hand:

    >>> for i in range(6):
    ...     results = model.fit(data['Y'])
    ...     print "AR coefficients:", model.rho
    ...     rho, sigma = yule_walker(data["Y"] - results.predicted,
    ...                              order=2,
    ...                              df=model.df_resid)
    ...     model = ARModel(model.design, rho) #doctest: +FP_6DP
    ...
    AR coefficients: [ 0.  0.]
    AR coefficients: [-0.61530877 -1.01542645]
    AR coefficients: [-0.72660832 -1.06201457]
    AR coefficients: [-0.7220361  -1.05365352]
    AR coefficients: [-0.72229201 -1.05408193]
    AR coefficients: [-0.722278   -1.05405838]
    >>> results.theta #doctest: +FP_6DP
    array([ 1.59564228, -0.58562172])
    >>> results.t() #doctest: +FP_6DP
    array([ 38.0890515 ,  -3.45429252])
    >>> print results.Tcontrast([0,1]) #doctest: +FP_6DP
    <T contrast: effect=-0.58562172384377043, sd=0.16953449108110835, t=-3.4542925165805847, df_den=5>
    >>> print results.Fcontrast(np.identity(2)) #doctest: +FP_6DP
    <F contrast: F=4216.810299725842, df_den=5, df_num=2>

    Reinitialize the model, and do the automated iterative fit

    >>> model.rho = np.array([0,0])
    >>> model.iterative_fit(data['Y'], niter=3)
    >>> print model.rho #doctest: +FP_6DP
    [-0.7220361  -1.05365352]
    """
    def __init__(self, design, rho):
        """ Initialize AR model instance

        Parameters
        ----------
        design : ndarray
            2D array with design matrix
        rho : int or array-like
            If int, gives order of model, and initializes rho to zeros.  If
            ndarray, gives initial estimate of rho. Be careful as ``ARModel(X,
            1) != ARModel(X, 1.0)``.
        """
        if type(rho) is type(1):
            self.order = rho
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0,1]:
                raise ValueError("AR parameters must be a scalar or a vector")
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        super(ARModel, self).__init__(design)

    def iterative_fit(self, Y, niter=3):
        """
        Perform an iterative two-stage procedure to estimate AR(p)
        parameters and regression coefficients simultaneously.

        Parameters
        ----------
        Y : ndarray
            data to which to fit model
        niter : optional, int
            the number of iterations (default 3)

        Returns
        -------
        None
        """
        for i in range(niter):
            self.initialize(self.design)
            results = self.fit(Y)
            self.rho, _ = yule_walker(Y - results.predicted,
                                      order=self.order, df=self.df_resid)

    def whiten(self, X):
        """ Whiten a series of columns according to AR(p) covariance structure

        Parameters
        ----------
        X : array-like
            array to whiten

        Returns
        -------
        wX : ndarray
            X whitened with order self.order AR
        """
        X = np.asarray(X, np.float64)
        _X = X.copy()
        for i in range(self.order):
            _X[(i+1):] = _X[(i+1):] - self.rho[i] * X[0:-(i+1)]
        return _X


def yule_walker(X, order=1, method="unbiased", df=None, inv=False):
    """
    Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    unbiased or maximum-likelihood estimator (mle)

    See, for example:

    http://en.wikipedia.org/wiki/Autoregressive_moving_average_model

    Parameters
    ----------
    X : (N,) ndarray
    order : int, optional
        Order of AR process.
    method : str, optional
        Method can be "unbiased" or "mle" and this determines denominator in
        estimate of autocorrelation function (ACF) at lag k. If "mle", the
        denominator is N=X.shape[0], if "unbiased" the denominator is N-k.
    df : int, optional
        Specifies the degrees of freedom. If df is supplied, then it is assumed
        the X has df degrees of freedom rather than N.
    inv : bool, optional
        Whether to return the inverse of the R matrix (see code)

    Returns
    -------
    rho : (`order`,) ndarray
    sigma : int
        standard deviation of the residuals after fit
    R_inv : ndarray
        If `inv` is True, also return the inverse of the R matrix

    Notes
    -----
    See also
    http://en.wikipedia.org/wiki/AR_model#Calculation_of_the_AR_parameters
    """
    method = str(method).lower()
    if method not in ["unbiased", "mle"]:
        raise ValueError("ACF estimation method must be 'unbiased or 'MLE'")
    X = np.asarray(X, np.float64)
    if X.ndim != 1:
        raise ValueError("Expecting a vector to estimate AR parameters")
    X -= X.mean(0)
    n = df or X.shape[0]
    if method == "unbiased":
        den = lambda k: n - k
    else:
        den = lambda k: n
    r = np.zeros(order+1, np.float64)
    r[0] = (X**2).sum() / den(0)
    for k in range(1,order+1):
        r[k] = (X[0:-k]*X[k:]).sum() / den(k)
    R = spl.toeplitz(r[:-1])
    rho = spl.solve(R, r[1:])
    sigmasq = r[0] - (r[1:]*rho).sum()
    if inv == True:
        return rho, np.sqrt(sigmasq), spl.inv(R)
    return rho, np.sqrt(sigmasq)


class WLSModel(OLSModel):
    """
    A regression model with diagonal but non-identity covariance structure.

    The weights are presumed to be (proportional to the) inverse of the variance
    of the observations.

    Examples
    --------
    >>> from nipy.modalities.fmri.formula import Term, Formula
    >>> data = np.rec.fromarrays(([1,3,4,5,2,3,4], range(1,8)), names=('Y', 'X'))
    >>> f = Formula([Term("X"), 1])
    >>> dmtx = f.design(data, return_float=True)
    >>> model = WLSModel(dmtx, weights=range(1,8))
    >>> results = model.fit(data['Y'])
    >>> results.theta
    array([ 0.0952381 ,  2.91666667])
    >>> results.t()
    array([ 0.35684428,  2.0652652 ])
    >>> print results.Tcontrast([0,1]) #doctest: +FP_6DP
    <T contrast: effect=2.91666666667, sd=1.41224801095, t=2.06526519708, df_den=5>
    >>> print results.Fcontrast(np.identity(2)) #doctest: +FP_6DP
    <F contrast: F=26.9986072423, df_den=5, df_num=2>
    """
    def __init__(self, design, weights=1):
        weights = np.array(weights)
        if weights.shape == (): # scalar
            self.weights = weights
        else:
            design_rows = design.shape[0]
            if not(weights.shape[0] == design_rows and
                   weights.size == design_rows) :
                raise ValueError(
                    'Weights must be scalar or same length as design')
            self.weights = weights.reshape(design_rows)
        super(WLSModel, self).__init__(design)

    def whiten(self, X):
        """
        Whitener for WLS model, multiplies by sqrt(self.weights)
        """
        X = np.asarray(X, np.float64)

        if X.ndim == 1:
            return X * np.sqrt(self.weights)
        elif X.ndim == 2:
            c = np.sqrt(self.weights)
            v = np.zeros(X.shape, np.float64)
            for i in range(X.shape[1]):
                v[:,i] = X[:,i] * c
            return v


class RegressionResults(LikelihoodModelResults):
    """
    This class summarizes the fit of a linear regression model.

    It handles the output of contrasts, estimates of covariance, etc.
    """
    def __init__(self, theta, Y, model, wY, wresid, cov=None, dispersion=1., nuisance=None):
        """
        See LikelihoodModelResults constructor.

        The only difference is that the whitened Y and residual values are stored for
        a regression model.
        """
        LikelihoodModelResults.__init__(self, theta, Y, model, cov, dispersion, nuisance)
        self.wY = wY
        self.wresid = wresid

    @setattr_on_read
    def resid(self):
        """
        Residuals from the fit.
        """
        return self.Y - self.predicted

    @setattr_on_read
    def norm_resid(self):
        """
        Residuals, normalized to have unit length.

        Notes
        -----
        Is this supposed to return "stanardized residuals," residuals standardized
        to have mean zero and approximately unit variance?

        d_i = e_i/sqrt(MS_E)

        Where MS_E = SSE/(n - k)

        See: Montgomery and Peck 3.2.1 p. 68
             Davidson and MacKinnon 15.2 p 662
        """
        return self.resid * utils.pos_recipr(np.sqrt(self.dispersion))

# predict is a verb
# do the predicted values need to be done automatically, then?
# or should you give a predict method similar to STATA

    @setattr_on_read
    def predicted(self):
        """
        Return linear predictor values from a design matrix.
        """
        beta = self.theta # the LikelihoodModelResults has parameters named 'theta'
        X = self.model.design
        return np.dot(X, beta)

    @setattr_on_read 
    def R2_adj(self):
        """
        Return the R^2 value for each row of the response Y.

        Notes
        -----
        Changed to the textbook definition of R^2.

        See: Davidson and MacKinnon p 74
        """
        if not self.model.has_intercept:
            warnings.warn("model does not have intercept term, SST inappropriate")
        d = 1. - self.R2
        d *= ((self.df_total - 1.) / self.df_resid)
        return 1 - d

    @setattr_on_read 
    def R2(self):
        """
        Return the adjusted R^2 value for each row of the response Y.

        Notes
        -----
        Changed to the textbook definition of R^2.

        See: Davidson and MacKinnon p 74
        """
        d = self.SSE / self.SST
        return 1 - d

    @setattr_on_read
    def SST(self):
        """
        Total sum of squares. If not from an OLS model this is "pseudo"-SST.
        """
        if not self.model.has_intercept:
            warnings.warn("model does not have intercept term, SST inappropriate")
        return ((self.wY - self.wY.mean(0))**2).sum(0)

    @setattr_on_read
    def SSE(self):
        """
        Error sum of squares. If not from an OLS model this is "pseudo"-SSE.
        """
        return (self.wresid**2).sum(0)

    @setattr_on_read
    def SSR(self):
        """
        Regression sum of squares
        """
        return self.SST - self.SSE

    @setattr_on_read
    def MSR(self):
        """
        Mean square (regression)
        """
        return self.SSR / (self.df_model - 1)

    @setattr_on_read
    def MSE(self):
        """
        Mean square (error)
        """
        return self.SSE / self.df_resid

    @setattr_on_read
    def MST(self):
        """
        Mean square (total)
        """
        return self.SST / (self.df_total - 1)

    @setattr_on_read
    def F_overall(self):
        """
        Overall goodness of fit F test, comparing model
        to a model with just an intercept. If not an OLS
        model this is a pseudo-F.
        """
        F = self.MSR / self.MSE
        Fp = stats.f.sf(F, self.df_model - 1, self.df_resid)
        return {'F':F, 'p_value':Fp, 'df_num': self.df_model-1, 'df_den': self.df_resid}


class GLSModel(OLSModel):
    """
    Generalized least squares model with a general covariance structure
    """
    def __init__(self, design, sigma):
        self.cholsigmainv = spl.linalg.cholesky(spl.linalg.pinv(sigma)).T
        super(GLSModel, self).__init__(design)

    def whiten(self, Y):
        return np.dot(self.cholsigmainv, Y)


def isestimable(C, D):
    """
    From an q x p contrast matrix C and an n x p design matrix D, checks
    if the contrast C is estimable by looking at the rank of vstack([C,D]) and
    verifying it is the same as the rank of D.
    """
    if C.ndim == 1:
        C.shape = (C.shape[0], 1)
    new = np.vstack([C, D])
    if utils.rank(new) != utils.rank(D):
        return False
    return True
