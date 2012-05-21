# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module presents an interface to use the glm implemented in
nipy.algorithms.statistics.models.regression
It contains the GLM and contrast classes that are meant to be the main objects
of fMRI data analyses.

It is important to note that the GLM is meant as a one-session 
General Linear Model. But inference can be performed on multiple sessions 
by computing fixed effects on contrasts


>>> from nipy.modalities.fmri.glm import GLM
>>> import numpy as np
>>> n, p, q = 100, 80, 10
>>> X, Y = np.random.randn(p, q), np.random.randn(p, n)
>>> GLMResult = GLM(X).fit(Y)
>>> cval = np.hstack((1, np.ones(9)))
>>> z_vals = GLMResult.contrast(cval).z_score()
>>> print z_vals.mean(), z_vals.std()

fixme
-----
a bug in mahalanobis
"""

import numpy as np
import scipy.stats as sps
from nipy.algorithms.statistics.models.regression import OLSModel, ARModel
from nipy.labs.utils import mahalanobis
from nipy.labs.utils.zscore import zscore

DEF_TINY = 1e-50
DEF_DOFMAX = 1e10


def glm_fit(X, Y, model='ar1', steps=100):
    """ GLM fitting of a dataset using 'ols' regression or the two-pass 

    Parameters
    ----------
    X: array of shape(n_time_points, n_regressors),  
       the design matrix
    Y: array of shape(n_time_points, n_samples), the fMRI data
    model: string, to be chosen in ['ar1', 'ols'], optional, 
           the temporal variance model. Defaults to 'ar1' 
    steps: int, optional,
           Maximum number of discrete steps for the AR(1) coef histogram

    Returns
    -------
    result: a GLMResults instance yielding the result of the GLM 
    """
    if model not in ['ar1', 'ols']:
        raise ValueError('Unknown model')

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    if Y.shape[0] != X.shape[0]:
        raise ValueError('Response and predictors are inconsistent')
        
    # fit the OLS model
    ols_result = OLSModel(X).fit(Y)

    # compute and discretize the AR1 coefs
    ar1 = ((ols_result.resid[1:] * ols_result.resid[:-1]).sum(0) /
           (ols_result.resid ** 2).sum(0))
    ar1 = (ar1 * steps).astype(np.int) * 1. / steps

    # Fit the AR model acccording to current AR(1) estimates
    if model == 'ar1':
        results_ = {}
        labels_ = ar1

        # fit the model
        for val in np.unique(labels_):
            m = ARModel(X, val)
            results_[val] = m.fit(Y[:, labels_ == val])
    else:
        # save the results
        labels_ = np.zeros(Y.shape[1])
        results_ = {0.0: ols_result}
    return GLMResults(results_, labels_)


class GLMResults(object):
    """ This class contains the dump of a GLM, in an efficient way
    It is meant to spit out a set of student or Fisher when given a contrast.
    In practice this hide loops over set of voxels
    """

    def __init__(self, glm_results, labels):
        """
        Parameters
        ----------
        glm_results: dictionary of  nipy.fixes.scipy.stats.models.regression,
                     describing results of a GLM fit
        labels: array of shape(n_voxels), 
                labels that associate each voxel with a results key
        """
        if glm_results.keys().sort() != labels.copy().sort():
            raise ValueError('results and labels are inconsistant')
        self.results = glm_results
        self.labels = labels
        # fixme: check that the number of voxels corresponds 
        # to the number of items in glm_results

    def contrast(self, c, contrast_type=None):
        """ Specify and estimate a linear contrast
        
        Parameters
        ----------
        c: numpy.ndarray of shape (p) or (q, p),
           where q = number of contrast vectors and p = number of regressors
        contrast_type: string, optional, either 't' or 'F', 
                       type of the contrast
        
        Returns
        -------
        con: Contrast instance
        """
        c = np.asarray(c)
        if c.ndim == 1:
            dim = 1
        else:
            dim = c.shape[0]
        if contrast_type is None:
            if dim == 1:
                contrast_type = 't'
            else:
                contrast_type = 'F'
        if contrast_type not in ['t', 'F']:
            raise ValueError('Unknown contrast type: %s' % contrast_type)
        
        effect_ = np.zeros((dim, self.labels.size), dtype=np.float)
        var_ = np.zeros((dim, dim, self.labels.size), dtype=np.float)
        #stat_ = np.zeros(self.labels.size, dtype=np.float)
        if contrast_type == 't':
            for l in self.results.keys():
                resl = self.results[l].Tcontrast(c)
                effect_[:, self.labels == l] = resl.effect.T
                var_[:, :, self.labels == l] = (resl.sd ** 2).T
                #stat_[self.labels == l] = resl.t
        else:
            for l in self.results.keys():
                resl = self.results[l].Fcontrast(c)
                effect_[:, self.labels == l] = resl.effect
                var_[:, :, self.labels == l] = resl.covariance
        con = Contrast(dim, contrast_type)
        con.effect = effect_
        con.variance = var_
        con.dof = self.results[l].df_resid
        return con


class Contrast(object):
    """ The contrast class handles the estimation of statistical contrasts
    After application of the GLM.
    The important feature is that it supports addition, 
    thus opening the possibility of fixed-effects models.
    
    The current implementation is meant to be simple, 
    and could be enhanced in the future on the computational side 
    (high-dimensional F constrasts may lead to memory breakage)
    """

    def __init__(self, dim=1, contrast_type='t', tiny=DEF_TINY, 
                 dofmax=DEF_DOFMAX):
        """
        Parameters
        ==========
        dim: int, the dimension of the contrast
        """
        self.dim = dim
        self.effect, self.variance, self.dof = None, None, None
        if dim > 1 and contrast_type is 't':
            print 'Automatically converted multi-dimensional t to F contrast'
            contrast_type = 'F'
        self.contrast_type = contrast_type
        self.stat_ = None
        self.p_value_ = None
        self.baseline = 0
        self.tiny = tiny
        self.dofmax = dofmax

    def stat(self, baseline=0.0):
        """ Return the decision statistic associated with the test of the
        null hypothesis: (H0) 'contrast equals baseline'
        
        Parameters
        ==========
        baseline: float, optional,
                  Baseline value for the test statistic
        """
        self.baseline = baseline

        # Case: one-dimensional contrast ==> t or t**2
        if self.dim == 1:
            # avoids division by zero
            stat = (self.effect - baseline) / np.sqrt(
                np.maximum(self.variance, self.tiny))
            if self.contrast_type == 'F':
                stat = stat ** 2
        # Case: F contrast
        elif self.contrast_type == 'F':
            # F = |t|^2/q ,  |t|^2 = e^t inv(v) e
            if self.effect.ndim == 1:
                self.effect = self.effect[np.newaxis]
            if self.variance.ndim == 1:
                self.variance = self.variance[np.newaxis, np.newaxis]
            stat = mahalanobis(self.effect - baseline, np.maximum(
                    self.variance, self.tiny)) / self.dim
        # Case: tmin (conjunctions)
        elif self.contrast_type == 'tmin':
            vdiag = self.variance.reshape([self.dim ** 2] + list(
                    self.variance.shape[2:]))[:: self.dim + 1]
            stat = (self.effect - baseline) / np.sqrt(
                np.maximum(vdiag, self.tiny))
            stat = stat.min(0)

        # Unknwon stat
        else:
            raise ValueError('Unknown statistic type')
        self.stat_ = stat
        return stat.ravel()

    def p_value(self, baseline=0.0):
        """Return a parametric estimate of the p-value associated
        with the null hypothesis: (H0) 'contrast equals baseline'

        Parameters
        ==========
        baseline: float, optional,
        Baseline value for the test statistic
        """
        if self.stat_ == None or not self.baseline == baseline:
            self.stat_ = self.stat(baseline)
        # Valid conjunction as in Nichols et al, Neuroimage 25, 2005.
        if self.contrast_type in ['t', 'tmin']:
            p = sps.t.sf(self.stat_, np.minimum(self.dof, self.dofmax))
        elif self.contrast_type == 'F':
            p = sps.f.sf(self.stat_, self.dim, np.minimum(
                    self.dof, self.dofmax))
        else:
            raise ValueError('Unknown statistic type')
        self.p_value_ = p
        return p

    def z_score(self, baseline=0.0):
        """Return a parametric estimation of the z-score associated
        with the null hypothesis: (H0) 'contrast equals baseline'
        
        Parameters
        ==========
        baseline: float, optional,
                  Baseline value for the test statistic
        """
        if self.p_value_ == None or not self.baseline == baseline:
            self.p_value_ = self.p_value(baseline)

        # Avoid inf values kindly supplied by scipy.
        return zscore(self.p_value_)

    def __add__(self, other):
        if self.contrast_type != other.contrast_type:
            raise ValueError(
                'The two contrasts do not have consistant type dimensions')
        if self.dim != other.dim:
            raise ValueError(
                'The two contrasts do not have compatible dimensions')
        con = Contrast(self.dim)
        con.contrast_type = self.contrast_type
        con.effect = self.effect + other.effect
        con.variance = self.variance + other.variance
        con.dof = self.dof + other.dof
        return con

    def __rmul__(self, scalar):
        """Multiplication of the contrast by a scalar"""
        scalar = float(scalar)
        con = Contrast(self.dim)
        con.contrast_type = self.contrast_type
        con.effect = self.effect * scalar
        con.variance = self.variance * scalar ** 2 
        con.dof = self.dof
        return con

    __mul__ = __rmul__

    def __div__(self, scalar):
        return self.__rmul__(1 / float(scalar))
