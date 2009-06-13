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
from string import join as sjoin
from csv import reader

import numpy as np
from scipy.linalg import norm, toeplitz

from nipy.fixes.scipy.stats.models.model import LikelihoodModel, \
     LikelihoodModelResults
from nipy.fixes.scipy.stats.models import utils

from scipy import stats
from scipy.stats.stats import ss

from descriptors import setattr_on_read

import numpy.lib.recfunctions as nprf

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
    >>> import numpy as N
    >>>
    >>> from nipy.fixes.scipy.stats.models.formula import Term, I
    >>> from nipy.fixes.scipy.stats.models.regression import OLSModel
    >>>
    >>> data={'Y':[1,3,4,5,2,3,4],
    ...       'X':range(1,8)}
    >>> f = term("X") + I
    >>> f.namespace = data
    >>>
    >>> model = OLSModel(f.design())
    >>> results = model.fit(data['Y'])
    >>>
    >>> results.beta
    array([ 0.25      ,  2.14285714])
    >>> results.t()
    array([ 0.98019606,  1.87867287])
    >>> print results.Tcontrast([0,1])
    <T contrast: effect=2.14285714286, sd=1.14062281591, t=1.87867287326, df_denom=5>
    >>> print results.Fcontrast(np.identity(2))
    <F contrast: F=19.4607843137, df_denom=5, df_num=2>
    """

    def __init__(self, design, hascons=True):
        super(OLSModel, self).__init__()
        self.initialize(design, hascons)

    def initialize(self, design, hascons=True):
# Jonathan: PLEASE don't assume we have a constant...
# TODO: handle case for noconstant regression
        self.design = design
        self.wdesign = self.whiten(self.design)
        self.calc_beta = np.linalg.pinv(self.wdesign)
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
            sigmasq = SSE / self.n
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

    def information(self, beta, Y, nuisance=None):
        # Jonathan: this is overwriting an abstract method of LikelihoodModel
        '''
        Returns the information matrix at (beta, Y, nuisance).
        
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
        The information matrix, the negative of the inverse of the
        Hessian of the
        of the log-likelihood function evaluated at (theta, Y, nuisance).
        

        '''
        X = self.design
        wY = self.whiten(Y)
        r = wY - np.dot(X, beta)
        n = self.df_total
        if nuisance is None:
            SSE = (r**2).sum(0)
            sigmasq = SSE / n
        else:
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
        ohat = np.dot(self.wdesign, self.calc_beta)
        if np.allclose(ohat, o):
            return True
        return False

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
        --------
        fit : RegressionResults
        """
        wY = self.whiten(Y)
        beta = np.dot(self.calc_beta, wY)
        wresid = wY - np.dot(self.wdesign, beta)
        dispersion = np.sum(wresid**2, 0) / (self.wdesign.shape[0] - self.wdesign.shape[1])
        lfit = RegressionResults(np.dot(self.calc_beta, wY), Y,
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
    >>> import numpy as N
    >>> import numpy.random as R
    >>>
    >>> from nipy.fixes.scipy.stats.models.formula import Term, I
    >>> from nipy.fixes.scipy.stats.models.regression import ARModel
    >>>
    >>> data={'Y':[1,3,4,5,8,10,9],
    ...       'X':range(1,8)}
    >>> f = term("X") + I
    >>> f.namespace = data
    >>>
    >>> model = ARModel(f.design(), 2)
    >>> for i in range(6):
    ...     results = model.fit(data['Y'])
    ...     print "AR coefficients:", model.rho
    ...     rho, sigma = model.yule_walker(data["Y"] - results.predict)
    ...     model = ARModel(model.design, rho)
    ...
    AR coefficients: [ 0.  0.]
    AR coefficients: [-0.52571491 -0.84496178]
    AR coefficients: [-0.620642   -0.88654567]
    AR coefficients: [-0.61887622 -0.88137957]
    AR coefficients: [-0.61894058 -0.88152761]
    AR coefficients: [-0.61893842 -0.88152263]
    >>> results.beta
    array([ 1.58747943, -0.56145497])
    >>> results.t()
    array([ 30.796394  ,  -2.66543144])
    >>> print results.Tcontrast([0,1])
    <T contrast: effect=-0.561454972239, sd=0.210643186553, t=-2.66543144085, df_denom=5>
    >>> print results.Fcontrast(np.identity(2))
    <F contrast: F=2762.42812716, df_denom=5, df_num=2>
    >>>
    >>> model.rho = np.array([0,0])
    >>> model.iterative_fit(data['Y'], niter=3)
    >>> print model.rho
    [-0.61887622 -0.88137957]
    """
    def __init__(self, design, rho):
        if type(rho) is type(1):
            self.order = rho
            self.rho = np.zeros(self.order, np.float64)
        else:
            self.rho = np.squeeze(np.asarray(rho))
            if len(self.rho.shape) not in [0,1]:
                raise ValueError, "AR parameters must be a scalar or a vector"
            if self.rho.shape == ():
                self.rho.shape = (1,)
            self.order = self.rho.shape[0]
        super(ARModel, self).__init__(design)

    def iterative_fit(self, Y, niter=3):
        """
        Perform an iterative two-stage procedure to estimate AR(p)
        parameters and regression coefficients simultaneously.

        :Parameters:
            Y : TODO
                TODO
            niter : ``integer``
                the number of iterations
        """
        for i in range(niter):
            self.initialize(self.design)
            results = self.fit(Y)
            self.rho, _ = yule_walker(Y - results.predict,
                                      order=self.order, df=self.df)

    def whiten(self, X):
        """
        Whiten a series of columns according to an AR(p)
        covariance structure.

        :Parameters:
            X : TODO
                TODO
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

    :Parameters:
        X : a 1d ndarray
        method : ``string``
               Method can be "unbiased" or "mle" and this determines
               denominator in estimate of autocorrelation function (ACF)
               at lag k. If "mle", the denominator is n=r.shape[0], if
               "unbiased" the denominator is n-k.
        df : ``integer``
               Specifies the degrees of freedom. If df is supplied,
               then it is assumed the X has df degrees of
               freedom rather than n.
    """

    method = str(method).lower()
    if method not in ["unbiased", "mle"]:
        raise ValueError, "ACF estimation method must be 'unbiased' \
        or 'MLE'"
    X = np.asarray(X, np.float64)
    X -= X.mean(0)
    n = df or X.shape[0]

    if method == "unbiased":
        denom = lambda k: n - k
    else:
        denom = lambda k: n

    if len(X.shape) != 1:
        raise ValueError, "expecting a vector to estimate AR parameters"
    r = np.zeros(order+1, np.float64)
    r[0] = (X**2).sum() / denom(0)
    for k in range(1,order+1):
        r[k] = (X[0:-k]*X[k:]).sum() / denom(k)
    R = toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    sigmasq = r[0] - (r[1:]*rho).sum()
    if inv == True:
        return rho, np.sqrt(sigmasq), np.linalg.inv(R)
    else:
        return rho, np.sqrt(sigmasq)

class WLSModel(OLSModel):
    """
    A regression model with diagonal but non-identity covariance
    structure. The weights are presumed to be
    (proportional to the) inverse of the
    variance of the observations.

    >>> import numpy as N
    >>>
    >>> from nipy.fixes.scipy.stats.models.formula import Term, I
    >>> from nipy.fixes.scipy.stats.models.regression import WLSModel
    >>>
    >>> data={'Y':[1,3,4,5,2,3,4],
    ...       'X':range(1,8)}
    >>> f = term("X") + I
    >>> f.namespace = data
    >>>
    >>> model = WLSModel(f.design(), weights=range(1,8))
    >>> results = model.fit(data['Y'])
    >>>
    >>> results.beta
    array([ 0.0952381 ,  2.91666667])
    >>> results.t()
    array([ 0.35684428,  2.0652652 ])
    >>> print results.Tcontrast([0,1])
    <T contrast: effect=2.91666666667, sd=1.41224801095, t=2.06526519708, df_denom=5>
    >>> print results.Fcontrast(np.identity(2))
    <F contrast: F=26.9986072423, df_denom=5, df_num=2>
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
        beta = self.theta # the LikelihoodModelResults has parameters named 'theta'
        X = self.model.design
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

        return self.resid * utils.recipr(np.sqrt(self.dispersion))

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
    def R2(self):
        """
        Return the R^2 value for each row of the response Y.
        
        Notes
        -----
        Changed to the textbook definition of R^2.
        
        See: Davidson and MacKinnon p 74
        """
        if not self.model.has_intercept:
            warnings.warn("model does not have intercept term, SST inappropriate")
        d = 1 - self.R2_adj
        d *= ((self.df_total - 1) / self.df_resid)
        return 1 - d

    @setattr_on_read
    def SST(self):
        """
        Total sum of squares. If not from an OLS model
        this is "pseudo"-SST.
        """
        if not self.model.has_intercept:
            warnings.warn("model does not have intercept term, SST inappropriate")
        return np.std(self.wY,axis=0)**2

    @setattr_on_read
    def SSE(self):
        """
        Error sum of squares. If not from an OLS model
        this is "pseudo"-SSE.
        """
        return np.sum(self.wresid**2, 0)

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
        return self.SSR / self.df_model

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
        Fp = stats.f.pdf(F, self.df_model, self.df_resid)
        return {'F':F, 'p-value':Fp, 'df_num': self.df_model, 'df_den': self.df_resid}

    @setattr_on_read 
    def R2_adj(self):
        """
        Return the adjusted R^2 value for each row of the response Y.
        
        Notes
        -----
        Changed to the textbook definition of R^2.
        
        See: Davidson and MacKinnon p 74
        """
        d = self.SSE / self.SST
        return 1 - d

class GLSModel(OLSModel):

    """
    Generalized least squares model with a general covariance structure

    This should probably go into nipy.fixes.scipy.stats.models.regression

    """

    def __init__(self, design, sigma):
        self.cholsigmainv = np.linalg.cholesky(np.linalg.pinv(sigma)).T
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
