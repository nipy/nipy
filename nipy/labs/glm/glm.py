# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import scipy.stats as sps

from . import kalman
from ..utils import mahalanobis
from ..utils.zscore import zscore


DEF_TINY = 1e-50
DEF_DOFMAX = 1e10
models = {'spherical': ['ols', 'kalman'], 'ar1': ['kalman']}


class glm(object):

    def __init__(self, Y=None, X=None, formula=None, axis=0,
             model='spherical', method=None, niter=2):

        # Check dimensions
        if Y == None:
            return
        else:
            self.fit(Y, X, formula, axis, model, method, niter)

    def fit(self, Y, X, formula=None, axis=0, model='spherical', method=None,
            niter=2):

        if Y.shape[axis] != X.shape[0]:
            raise ValueError('Response and predictors are inconsistent')

        # Find model type
        self._axis = axis
        if isinstance(formula, str):
            model = 'mfx'
        if model in models:
            self.model = model
            if method == None:
                self.method = models[model][0]
            elif models[model].count(method):
                self.method = method
            else:
                raise ValueError('Unknown method')
        else:
            raise ValueError('Unknown model')

        # Initialize fields
        constants = []
        a = 0

        # Switch on models / methods
        if self.model == 'spherical':
            constants = ['nvbeta', 'a']
            if self.method == 'ols':
                out = ols(Y, X, axis=axis)
            elif self.method == 'kalman':
                out = kalman.ols(Y, X, axis=axis)
        elif self.model == 'ar1':
            constants = ['a']
            out = kalman.ar1(Y, X, axis=axis, niter=niter)
            a = out[4]
            out = out[0: 4]

        # Finalize
        self.beta, self.nvbeta, self.s2, self.dof = out
        self.s2 = self.s2.squeeze()
        self.a = a
        self._constants = constants

    def save(self, file):
        """ Save fit into a .npz file
        """
        np.savez(file,
             beta=self.beta,
             nvbeta=self.nvbeta,
             s2=self.s2,
             dof=self.dof,
             a=self.a,
             model=self.model,
             method=self.method,
             axis=self._axis,
             constants=self._constants)

    def contrast(self, c, type='t', tiny=DEF_TINY, dofmax=DEF_DOFMAX):
        """ Specify and estimate a constrast

        c must be a numpy.ndarray (or anything that numpy.asarray
        can cast to a ndarray).
        For a F contrast, c must be q x p
        where q is the number of contrast vectors and
        p is the total number of regressors.
        """
        c = np.asarray(c)
        #dim = len(c.shape)
        if c.ndim == 1:
            dim = 1
        else:
            dim = c.shape[0]
        axis = self._axis
        ndims = len(self.beta.shape)

        # Compute the contrast estimate: c*B
        B = np.rollaxis(self.beta, axis, ndims)
        con = np.inner(c, B) # shape = q, X

        # Compute the variance of the contrast estimate: s2 * (c' * nvbeta * c)
        # Two cases are considered: either the input effect variance
        # is position-dependent (output by RKF_fit), or it is a global
        # one (output by KF_fit)
        s2 = self.s2.squeeze()
        nvbeta = self.nvbeta
        if not 'nvbeta' in self._constants:
            nvbeta = np.rollaxis(nvbeta, axis, ndims + 1)
            nvbeta = np.rollaxis(nvbeta, axis, ndims + 1) # shape = X, p, p
        if dim == 1:
            vcon = np.inner(c, np.inner(c, nvbeta))
            vcon = vcon.squeeze() * s2
        else:
            vcon = np.dot(c, np.inner(nvbeta, c)) # q, X, q or q, q
            if not 'nvbeta' in self._constants:
                vcon = np.rollaxis(vcon, ndims, 1) * s2 # q, q, X
            else:
                aux = vcon.shape # q, q
                vcon = np.resize(vcon, s2.shape + aux) # X, q, q
                vcon = vcon.T.reshape(aux + (s2.size,)) * \
                    s2.reshape((s2.size,)) # q, q, Xflat
                vcon = vcon.reshape(aux + s2.shape) # q, q, X

        # Create contrast instance
        c = contrast(dim, type, tiny, dofmax)
        c.effect = con
        c.variance = vcon
        c.dof = self.dof
        return c


class contrast(object):

    def __init__(self, dim, type='t', tiny=DEF_TINY, dofmax=DEF_DOFMAX):
        """tiny is a numerical constant for computations.
        """
        self.dim = dim
        self.effect = None
        self.variance = None
        self.dof = None
        if dim > 1:
            if type is 't':
                type = 'F'
        self.type = type
        self._stat = None
        self._pvalue = None
        self._baseline = 0
        self._tiny = tiny
        self._dofmax = dofmax

    def summary(self):
        """
        Return a dictionary containing the estimated contrast effect,
        the associated ReML-based estimation variance, and the estimated
        degrees of freedom (variance of the variance).
        """
        return {'effect': self.effect,
                'variance': self.variance,
                'dof': self.dof}

    def stat(self, baseline=0.0):
        """
        Return the decision statistic associated with the test of the
        null hypothesis: (H0) 'contrast equals baseline'
        """
        self._baseline = baseline

        # Case: one-dimensional contrast ==> t or t**2
        if self.dim == 1:
            # avoids division by zero
            t = (self.effect - baseline) / np.sqrt(
                np.maximum(self.variance, self._tiny))
            if self.type == 'F':
                t = t ** 2
        # Case: F contrast
        elif self.type == 'F':
            # F = |t|^2/q ,  |t|^2 = e^t v-1 e
            t = mahalanobis(self.effect - baseline, np.maximum(
                    self.variance, self._tiny)) / self.dim
        # Case: tmin (conjunctions)
        elif self.type == 'tmin':
            vdiag = self.variance.reshape([self.dim ** 2] + list(
                    self.variance.shape[2:]))[:: self.dim + 1]
            t = (self.effect - baseline) / np.sqrt(
                np.maximum(vdiag, self._tiny))
            t = t.min(0)

        # Unknwon stat
        else:
            raise ValueError('Unknown statistic type')
        self._stat = t
        return t

    def pvalue(self, baseline=0.0):
        """
        Return a parametric approximation of the p-value associated
        with the null hypothesis: (H0) 'contrast equals baseline'
        """
        if self._stat == None or not self._baseline == baseline:
            self._stat = self.stat(baseline)
        # Valid conjunction as in Nichols et al, Neuroimage 25, 2005.
        if self.type in ['t', 'tmin']:
            p = sps.t.sf(self._stat, np.minimum(self.dof, self._dofmax))
        elif self.type == 'F':
            p = sps.f.sf(self._stat, self.dim, np.minimum(
                    self.dof, self._dofmax))
        else:
            raise ValueError('Unknown statistic type')
        self._pvalue = p
        return p

    def zscore(self, baseline=0.0):
        """
        Return a parametric approximation of the z-score associated
        with the null hypothesis: (H0) 'contrast equals baseline'
        """
        if self._pvalue == None or not self._baseline == baseline:
            self._pvalue = self.pvalue(baseline)

        # Avoid inf values kindly supplied by scipy.
        z = zscore(self._pvalue)
        return z

    def __add__(self, other):
        if self.dim != other.dim:
            return None
        con = contrast(self.dim)
        con.type = self.type
        con.effect = self.effect + other.effect
        con.variance = self.variance + other.variance
        con.dof = self.dof + other.dof
        return con

    def __rmul__(self, other):
        k = float(other)
        con = contrast(self.dim)
        con.type = self.type
        con.effect = k * self.effect
        con.variance = k ** 2 * self.variance
        con.dof = self.dof
        return con

    __mul__ = __rmul__

    def __div__(self, other):
        return self.__rmul__(1 / float(other))


def ols(Y, X, axis=0):
    """Essentially, compute pinv(X)*Y
    """
    ndims = len(Y.shape)
    pX = np.linalg.pinv(X)
    beta = np.rollaxis(np.inner(pX, np.rollaxis(Y, axis, ndims)), 0, axis + 1)
    nvbeta = np.inner(pX, pX)
    res = Y - np.rollaxis(
        np.inner(X, np.rollaxis(beta, axis, ndims)), 0, axis + 1)
    n = res.shape[axis]
    s2 = (res ** 2).sum(axis) / float(n - X.shape[1])
    dof = float(X.shape[0] - X.shape[1])
    return beta, nvbeta, s2, dof


def load(file):
    """Load a fitted glm
    """
    from os.path import splitext
    if splitext(file)[1] == '':
        file = file + '.npz'
    fmod = np.load(file)
    mod = glm()
    mod.beta = fmod['beta']
    mod.nvbeta = fmod['nvbeta']
    mod.s2 = fmod['s2']
    mod.dof = fmod['dof']
    mod.a = fmod['a']
    mod.model = str(fmod['model'])
    mod.method = str(fmod['method'])
    mod._axis = int(fmod['axis'])
    mod._constants = list(fmod['constants'])
    return mod
