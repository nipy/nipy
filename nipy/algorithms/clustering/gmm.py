# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Gaussian Mixture Model Class:
contains the basic fields and methods of GMMs
The class GMM _old uses C bindings which are
computationally and memory efficient.

Author : Bertrand Thirion, 2006-2009
"""

import numpy as np
from scipy.linalg import eigvalsh


class GridDescriptor(object):
    """
    A tiny class to handle cartesian grids
    """

    def __init__(self, dim=1, lim=None, n_bins=None):
        """
        Parameters
        ----------
        dim: int, optional,
             the dimension of the grid
        lim: list of len(2*self.dim),
             the limits of the grid as (xmin, xmax, ymin, ymax, ...)
        n_bins: list of len(self.dim),
             the number of bins in each direction
        """
        self.dim = dim
        if lim is not None:
            self.set(lim, n_bins)
        if np.size(n_bins) == self.dim:
            self.n_bins = np.ravel(np.array(n_bins))

    def set(self, lim, n_bins=10):
        """ set the limits of the grid and the number of bins

        Parameters
        ----------
        lim: list of len(2*self.dim),
             the limits of the grid as (xmin, xmax, ymin, ymax, ...)
        n_bins: list of len(self.dim), optional
             the number of bins in each direction
        """
        if len(lim) == 2 * self.dim:
            self.lim = lim
        else:
            raise ValueError("Wrong dimension for grid definition")
        if np.size(n_bins) == self.dim:
            self.n_bins = np.ravel(np.array(n_bins))
        else:
            raise ValueError("Wrong dimension for grid definition")

    def make_grid(self):
        """ Compute the grid points

        Returns
        -------
        grid: array of shape (nb_nodes, self.dim)
              where nb_nodes is the prod of self.n_bins
        """
        size = np.prod(self.n_bins)
        grid = np.zeros((size, self.dim))
        grange = []

        for j in range(self.dim):
            xm = self.lim[2 * j]
            xM = self.lim[2 * j + 1]
            if np.isscalar(self.n_bins):
                xb = self.n_bins
            else:
                xb = self.n_bins[j]
            gr = xm + float(xM - xm) / (xb - 1) * np.arange(xb).astype('f')
            grange.append(gr)

        if self.dim == 1:
            grid = np.array([[grange[0][i]] for i in range(xb)])

        if self.dim == 2:
            for i in range(self.n_bins[0]):
                for j in range(self.n_bins[1]):
                    grid[i * self.n_bins[1] + j] = np.array(
                        [grange[0][i], grange[1][j]])

        if self.dim == 3:
            for i in range(self.n_bins[0]):
                for j in range(self.n_bins[1]):
                    for k in range(self.n_bins[2]):
                        q = (i * self.n_bins[1] + j) * self.n_bins[2] + k
                        grid[q] = np.array([grange[0][i], grange[1][j],
                                           grange[2][k]])
        if self.dim > 3:
            raise NotImplementedError(
                'only dimensions <4 are currently handled')
        return grid


def best_fitting_GMM(x, krange, prec_type='full', niter=100, delta=1.e-4,
                     ninit=1, verbose=0):
    """
    Given a certain dataset x, find the best-fitting GMM
    with a number k of classes in a certain range defined by krange

    Parameters
    ----------
    x: array of shape (n_samples,dim)
       the data from which the model is estimated
    krange: list of floats,
            the range of values to test for k
    prec_type: string (to be chosen within 'full','diag'), optional,
              the covariance parameterization
    niter: int, optional,
           maximal number of iterations in the estimation process
    delta: float, optional,
           increment of data likelihood at which convergence is declared
    ninit: int
           number of initialization performed
    verbose=0: verbosity mode

    Returns
    -------
    mg : the best-fitting GMM instance
    """
    if np.size(x) == x.shape[0]:
        x = np.reshape(x, (np.size(x), 1))

    dim = x.shape[1]
    bestbic = - np.inf
    for k in krange:
        lgmm = GMM(k, dim, prec_type)
        gmmk = lgmm.initialize_and_estimate(x, None, niter, delta, ninit,
                                            verbose)
        bic = gmmk.evidence(x)
        if bic > bestbic:
            bestbic = bic
            bgmm = gmmk
        if verbose:
            print 'k', k, 'bic', bic
    return bgmm


def plot2D(x, my_gmm, z=None, with_dots=True, log_scale=False, mpaxes=None,
           verbose=0):
    """
    Given a set of points in a plane and a GMM, plot them

    Parameters
    ----------
    x: array of shape (npoints, dim=2),
        sample points
    my_gmm: GMM instance,
            whose density has to be ploted
    z: array of shape (npoints), optional
       that gives a labelling of the points in x
       by default, it is not taken into account
    with_dots, bool, optional
               whether to plot the dots or not
    log_scale: bool, optional
               whether to plot the likelihood in log scale or not
    mpaxes=None, int, optional
                 if not None, axes handle for plotting
    verbose: verbosity mode, optional

    Returns
    -------
    gd, GridDescriptor instance,
        that represents the grid used in the function
    ax, handle to the figure axes

    Note
    ----
    my_gmm is assumed to have have a  'nixture_likelihood' method
    that takes an array of points of shape (np, dim)
    and returns an array of shape (np,my_gmm.k)
    that represents  the likelihood component-wise
    """
    import matplotlib.pyplot as plt

    if x.shape[1] != my_gmm.dim:
        raise ValueError('Incompatible dimension between data and model')
    if x.shape[1] != 2:
        raise ValueError('this works only for 2D cases')

    gd1 = GridDescriptor(2)
    xmin, xmax = x.min(0), x.max(0)
    xm = 1.1 * xmin[0] - 0.1 * xmax[0]
    xs = 1.1 * xmax[0] - 0.1 * xmin[0]
    ym = 1.1 * xmin[1] - 0.1 * xmax[1]
    ys = 1.1 * xmax[1] - 0.1 * xmin[1]

    gd1.set([xm, xs, ym, ys], [51, 51])
    grid = gd1.make_grid()
    L = my_gmm.mixture_likelihood(grid)
    if verbose:
        intl = L.sum() * (xs - xm) * (ys - ym) / 2500
        print 'integral of the density on the domain ', intl

    if mpaxes == None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)
    else:
        ax = mpaxes

    gdx = gd1.n_bins[0]
    Pdens = np.reshape(L, (gdx, np.size(L) / gdx))
    extent = [xm, xs, ym, ys]
    if log_scale:
        plt.imshow(np.log(Pdens.T), alpha=2.0, origin='lower',
                  extent=extent)
    else:
        plt.imshow(Pdens.T, alpha=2.0, origin='lower', extent=extent)

    if with_dots:
        if z == None:
            plt.plot(x[:, 0], x[:, 1], 'o')
        else:
            hsv = plt.cm.hsv(range(256))
            col = hsv[range(0, 256, 256 / int(z.max() + 1))]
            for k in range(z.max() + 1):
                plt.plot(x[z == k, 0], x[z == k, 1], 'o', color=col[k])

    plt.axis(extent)
    plt.colorbar()
    return gd1, ax


class GMM(object):
    """Standard GMM.

    this class contains the following members
    k (int): the number of components in the mixture
    dim (int): is the dimension of the data
    prec_type = 'full' (string) is the parameterization
              of the precisions/covariance matrices:
              either 'full' or 'diagonal'.
    means: array of shape (k,dim):
          all the means (mean parameters) of the components
    precisions: array of shape (k,dim,dim):
               the precisions (inverse covariance matrix) of the components
    weights: array of shape(k): weights of the mixture

    fixme
    -----
    no copy method
    """

    def __init__(self, k=1, dim=1, prec_type='full', means=None,
                 precisions=None, weights=None):
        """
        Initialize the structure, at least with the dimensions of the problem

        Parameters
        ----------
        k (int) the number of classes of the model
        dim (int) the dimension of the problem
        prec_type = 'full' : coavriance:precision parameterization
                  (diagonal 'diag' or full 'full').
        means = None: array of shape (self.k,self.dim)
        precisions = None:  array of shape (self.k,self.dim,self.dim)
                   or (self.k, self.dim)
        weights=None: array of shape (self.k)

        By default, means, precision and weights are set as
        zeros()
        eye()
        1/k ones()
        with the correct dimensions
        """
        self.k = k
        self.dim = dim
        self.prec_type = prec_type
        self.means = means
        self.precisions = precisions
        self.weights = weights

        if self.means == None:
            self.means = np.zeros((self.k, self.dim))

        if self.precisions == None:
            if prec_type == 'full':
                prec = np.reshape(np.eye(self.dim), (1, self.dim, self.dim))
                self.precisions = np.repeat(prec, self.k, 0)
            else:
                self.precisions = np.ones((self.k, self.dim))

        if self.weights == None:
            self.weights = np.ones(self.k) * 1.0 / self.k

    def plugin(self, means, precisions, weights):
        """
        Set manually the weights, means and precision of the model

        Parameters
        ----------
        means: array of shape (self.k,self.dim)
        precisions:  array of shape (self.k,self.dim,self.dim)
                     or (self.k, self.dim)
        weights: array of shape (self.k)
        """
        self.means = means
        self.precisions = precisions
        self.weights = weights
        self.check()

    def check(self):
        """
        Checking the shape of different matrices involved in the model
        """
        if self.means.shape[0] != self.k:
            raise ValueError("self.means does not have correct dimensions")

        if self.means.shape[1] != self.dim:
            raise ValueError("self.means does not have correct dimensions")

        if self.weights.size != self.k:
            raise ValueError("self.weights does not have correct dimensions")

        if self.dim != self.precisions.shape[1]:
            raise ValueError(
                "self.precisions does not have correct dimensions")

        if self.prec_type == 'full':
            if self.dim != self.precisions.shape[2]:
                raise ValueError(
                    "self.precisions does not have correct dimensions")

        if self.prec_type == 'diag':
            if np.shape(self.precisions) != np.shape(self.means):
                raise ValueError(
                    "self.precisions does not have correct dimensions")

        if self.precisions.shape[0] != self.k:
            raise ValueError(
                "self.precisions does not have correct dimensions")

        if self.prec_type not in ['full', 'diag']:
            raise ValueError('unknown precisions type')

    def check_x(self, x):
        """
        essentially check that x.shape[1]==self.dim

        x is returned with possibly reshaping
        """
        if np.size(x) == x.shape[0]:
            x = np.reshape(x, (np.size(x), 1))
        if x.shape[1] != self.dim:
            raise ValueError('incorrect size for x')
        return x

    def initialize(self, x):
        """Initializes self according to a certain dataset x:
        1. sets the regularizing hyper-parameters
        2. initializes z using a k-means algorithm, then
        3. upate the parameters

        Parameters
        ----------
        x, array of shape (n_samples,self.dim)
           the data used in the estimation process
        """
        from .utils import kmeans
        n = x.shape[0]

        #1. set the priors
        self.guess_regularizing(x, bcheck=1)

        # 2. initialize the memberships
        if self.k > 1:
            _, z, _ = kmeans(x, self.k)
        else:
            z = np.zeros(n).astype(np.int)

        l = np.zeros((n, self.k))
        l[np.arange(n), z] = 1

        # 3.update the parameters
        self.update(x, l)

    def pop(self, like, tiny=1.e-15):
        """compute the population, i.e. the statistics of allocation

        Parameters
        ----------
        like: array of shape (n_samples,self.k):
              the likelihood of each item being in each class
        """
        sl = np.maximum(tiny, np.sum(like, 1))
        nl = (like.T / sl).T
        return np.sum(nl, 0)

    def update(self, x, l):
        """ Identical to self._Mstep(x,l)
        """
        self._Mstep(x, l)

    def likelihood(self, x):
        """
        return the likelihood of the model for the data x
        the values are weighted by the components weights

        Parameters
        ----------
        x array of shape (n_samples,self.dim)
           the data used in the estimation process

        Returns
        -------
        like, array of shape(n_samples,self.k)
          component-wise likelihood
        """
        like = self.unweighted_likelihood(x)
        like *= self.weights
        return like

    def unweighted_likelihood_(self, x):
        """
        return the likelihood of each data for each component
        the values are not weighted by the component weights

        Parameters
        ----------
        x: array of shape (n_samples,self.dim)
           the data used in the estimation process

        Returns
        -------
        like, array of shape(n_samples,self.k)
          unweighted component-wise likelihood
        """
        n = x.shape[0]
        like = np.zeros((n, self.k))

        for k in range(self.k):
            # compute the data-independent factor first
            w = - np.log(2 * np.pi) * self.dim
            m = np.reshape(self.means[k], (1, self.dim))
            b = self.precisions[k]
            if self.prec_type == 'full':
                w += np.log(eigvalsh(b)).sum()
                dx = m - x
                q = np.sum(np.dot(dx, b) * dx, 1)
            else:
                w += np.sum(np.log(b))
                q = np.dot((m - x) ** 2, b)
            w -= q
            w /= 2
            like[:, k] = np.exp(w)
        return like

    def unweighted_likelihood(self, x):
        """
        return the likelihood of each data for each component
        the values are not weighted by the component weights

        Parameters
        ----------
        x: array of shape (n_samples,self.dim)
           the data used in the estimation process

        Returns
        -------
        like, array of shape(n_samples,self.k)
          unweighted component-wise likelihood

        Note
        ----
        Hopefully faster
        """
        xt = x.T.copy()
        n = x.shape[0]
        like = np.zeros((n, self.k))

        for k in range(self.k):
            # compute the data-independent factor first
            w = - np.log(2 * np.pi) * self.dim
            m = np.reshape(self.means[k], (self.dim, 1))
            b = self.precisions[k]
            if self.prec_type == 'full':
                w += np.log(eigvalsh(b)).sum()
                dx = xt - m
                sqx = dx * np.dot(b, dx)
                q = np.zeros(n)
                for d in range(self.dim):
                    q += sqx[d]
            else:
                w += np.sum(np.log(b))
                q = np.dot(b, (m - xt) ** 2)
            w -= q
            w /= 2
            like[:, k] = np.exp(w)
        return like

    def mixture_likelihood(self, x):
        """Returns the likelihood of the mixture for x

        Parameters
        ----------
        x: array of shape (n_samples,self.dim)
           the data used in the estimation process
        """
        x = self.check_x(x)
        like = self.likelihood(x)
        sl = np.sum(like, 1)
        return sl

    def average_log_like(self, x, tiny=1.e-15):
        """returns the averaged log-likelihood of the mode for the dataset x

        Parameters
        ----------
        x:  array of shape (n_samples,self.dim)
            the data used in the estimation process
        tiny = 1.e-15: a small constant to avoid numerical singularities
        """
        x = self.check_x(x)
        like = self.likelihood(x)
        sl = np.sum(like, 1)
        sl = np.maximum(sl, tiny)
        return np.mean(np.log(sl))

    def evidence(self, x):
        """Computation of bic approximation of evidence

        Parameters
        ----------
        x array of shape (n_samples,dim)
          the data from which bic is computed

        Returns
        -------
        the bic value
        """
        x = self.check_x(x)
        tiny = 1.e-15
        like = self.likelihood(x)
        return self.bic(like, tiny)

    def bic(self, like, tiny=1.e-15):
        """Computation of bic approximation of evidence

        Parameters
        ----------
        like, array of shape (n_samples, self.k)
           component-wise likelihood
        tiny=1.e-15, a small constant to avoid numerical singularities

        Returns
        -------
        the bic value, float
        """
        sl = np.sum(like, 1)
        sl = np.maximum(sl, tiny)
        bicc = np.sum(np.log(sl))

        # number of parameters
        n = like.shape[0]
        if self.prec_type == 'full':
            eta = self.k * (1 + self.dim + (self.dim * self.dim + 1) / 2) - 1
        else:
            eta = self.k * (1 + 2 * self.dim) - 1
        bicc = bicc - np.log(n) * eta
        return bicc

    def _Estep(self, x):
        """
        E step of the EM algo
        returns the likelihood per class of each data item

        Parameters
        ----------
        x array of shape (n_samples,dim)
          the data used in the estimation process

        Returns
        -------
        likelihood array of shape(n_samples,self.k)
                   component-wise likelihood
        """
        return self.likelihood(x)

    def guess_regularizing(self, x, bcheck=1):
        """
        Set the regularizing priors as weakly informative
        according to Fraley and raftery;
        Journal of Classification 24:155-181 (2007)

        Parameters
        ----------
        x array of shape (n_samples,dim)
          the data used in the estimation process
        """
        small = 0.01
        # the mean of the data
        mx = np.reshape(x.mean(0), (1, self.dim))

        dx = x - mx
        vx = np.dot(dx.T, dx) / x.shape[0]
        if self.prec_type == 'full':
            px = np.reshape(np.diag(1.0 / np.diag(vx)),
                            (1, self.dim, self.dim))
        else:
            px = np.reshape(1.0 / np.diag(vx), (1, self.dim))
        px *= np.exp(2.0 / self.dim * np.log(self.k))
        self.prior_means = np.repeat(mx, self.k, 0)
        self.prior_weights = np.ones(self.k) / self.k
        self.prior_scale = np.repeat(px, self.k, 0)
        self.prior_dof = self.dim + 2
        self.prior_shrinkage = small
        self.weights = np.ones(self.k) * 1.0 / self.k
        if bcheck:
            self.check()

    def _Mstep(self, x, like):
        """
        M step regularized according to the procedure of
        Fraley et al. 2007

        Parameters
        ----------
        x: array of shape(n_samples,self.dim)
           the data from which the model is estimated
        like: array of shape(n_samples,self.k)
           the likelihood of the data under each class
        """
        from numpy.linalg import pinv
        tiny = 1.e-15
        pop = self.pop(like)
        sl = np.maximum(tiny, np.sum(like, 1))
        like = (like.T / sl).T

        # shrinkage,weights,dof
        self.weights = self.prior_weights + pop
        self.weights = self.weights / self.weights.sum()

        # reshape
        pop = np.reshape(pop, (self.k, 1))
        prior_shrinkage = self.prior_shrinkage
        shrinkage = pop + prior_shrinkage

        # means
        means = np.dot(like.T, x) + self.prior_means * prior_shrinkage
        self.means = means / shrinkage

        #precisions
        empmeans = np.dot(like.T, x) / np.maximum(pop, tiny)
        empcov = np.zeros(np.shape(self.precisions))

        if self.prec_type == 'full':
            for k in range(self.k):
                dx = x - empmeans[k]
                empcov[k] = np.dot(dx.T, like[:, k:k + 1] * dx)
            #covariance
            covariance = np.array([pinv(self.prior_scale[k])
                                   for k in range(self.k)])
            covariance += empcov
            dx = np.reshape(empmeans - self.prior_means, (self.k, self.dim, 1))
            addcov = np.array([np.dot(dx[k], dx[k].T) for k in range(self.k)])
            apms = np.reshape(prior_shrinkage * pop / shrinkage,
                              (self.k, 1, 1))
            covariance += (addcov * apms)
            dof = self.prior_dof + pop + self.dim + 2
            covariance /= np.reshape(dof, (self.k, 1, 1))

            # precision
            self.precisions = np.array([pinv(covariance[k]) \
                                       for k in range(self.k)])
        else:
            for k in range(self.k):
                dx = x - empmeans[k]
                empcov[k] = np.sum(dx ** 2 * like[:, k:k + 1], 0)
            # covariance
            covariance = np.array([1.0 / self.prior_scale[k]
                                   for k in range(self.k)])
            covariance += empcov
            dx = np.reshape(empmeans - self.prior_means, (self.k, self.dim, 1))
            addcov = np.array([np.sum(dx[k] ** 2, 0) for k in range(self.k)])
            apms = np.reshape(prior_shrinkage * pop / shrinkage, (self.k, 1))
            covariance += addcov * apms
            dof = self.prior_dof + pop + self.dim + 2
            covariance /= np.reshape(dof, (self.k, 1))

            # precision
            self.precisions = np.array([1.0 / covariance[k] \
                                       for k in range(self.k)])

    def map_label(self, x, like=None):
        """return the MAP labelling of x

        Parameters
        ----------
        x array of shape (n_samples,dim)
          the data under study
        like=None array of shape(n_samples,self.k)
               component-wise likelihood
               if like==None, it is recomputed

        Returns
        -------
        z: array of shape(n_samples): the resulting MAP labelling
           of the rows of x
        """
        if like == None:
            like = self.likelihood(x)
        z = np.argmax(like, 1)
        return z

    def estimate(self, x, niter=100, delta=1.e-4, verbose=0):
        """ Estimation of the model given a dataset x

        Parameters
        ----------
        x array of shape (n_samples,dim)
          the data from which the model is estimated
        niter=100: maximal number of iterations in the estimation process
        delta = 1.e-4: increment of data likelihood at which
              convergence is declared
        verbose=0: verbosity mode

        Returns
        -------
        bic : an asymptotic approximation of model evidence
        """
        # check that the data is OK
        x = self.check_x(x)

        # alternation of E/M step until convergence
        tiny = 1.e-15
        av_ll_old = - np.inf
        for i in range(niter):
            l = self._Estep(x)
            av_ll = np.mean(np.log(np.maximum(np.sum(l, 1), tiny)))
            if av_ll < av_ll_old + delta:
                if verbose:
                    print 'iteration:', i, 'log-likelihood:', av_ll,\
                          'old value:', av_ll_old
                break
            else:
                av_ll_old = av_ll
            if verbose:
                print i, av_ll, self.bic(l)
            self._Mstep(x, l)

        return self.bic(l)

    def initialize_and_estimate(self, x, z=None, niter=100, delta=1.e-4,\
                                ninit=1, verbose=0):
        """Estimation of self given x

        Parameters
        ----------
        x array of shape (n_samples,dim)
          the data from which the model is estimated
        z = None: array of shape (n_samples)
            a prior labelling of the data to initialize the computation
        niter=100: maximal number of iterations in the estimation process
        delta = 1.e-4: increment of data likelihood at which
              convergence is declared
        ninit=1: number of initialization performed
                 to reach a good solution
        verbose=0: verbosity mode

        Returns
        -------
        the best model is returned
        """
        bestbic = - np.inf
        bestgmm = GMM(self.k, self.dim, self.prec_type)
        bestgmm.initialize(x)

        for i in range(ninit):
            # initialization -> Kmeans
            self.initialize(x)

            # alternation of E/M step until convergence
            bic = self.estimate(x, niter=niter, delta=delta, verbose=0)
            if bic > bestbic:
                bestbic = bic
                bestgmm.plugin(self.means, self.precisions, self.weights)

        return bestgmm

    def train(self, x, z=None, niter=100, delta=1.e-4, ninit=1, verbose=0):
        """Idem initialize_and_estimate
        """
        return self.initialize_and_estimate(x, z, niter, delta, ninit, verbose)

    def test(self, x, tiny=1.e-15):
        """Returns the log-likelihood of the mixture for x

        Parameters
        ----------
        x array of shape (n_samples,self.dim)
          the data used in the estimation process

        Returns
        -------
        ll: array of shape(n_samples)
            the log-likelihood of the rows of x
        """
        return np.log(np.maximum(self.mixture_likelihood(x), tiny))

    def show_components(self, x, gd, density=None, mpaxes=None):
        """Function to plot a GMM -- Currently, works only in 1D

        Parameters
        ----------
        x: array of shape(n_samples, dim)
           the data under study
        gd: GridDescriptor instance
        density: array os shape(prod(gd.n_bins))
                 density of the model one the discrete grid implied by gd
                 by default, this is recomputed
        mpaxes: axes handle to make the figure, optional,
                if None, a new figure is created
        """
        import matplotlib.pyplot as plt
        if density is None:
            density = self.mixture_likelihood(gd.make_grid())

        if gd.dim > 1:
            raise NotImplementedError("only implemented in 1D")

        step = 3.5 * np.std(x) / np.exp(np.log(np.size(x)) / 3)
        bins = max(10, int((x.max() - x.min()) / step))
        xmin = 1.1 * x.min() - 0.1 * x.max()
        xmax = 1.1 * x.max() - 0.1 * x.min()
        h, c = np.histogram(x, bins, [xmin, xmax], normed=True)

        # Make code robust to new and old behavior of np.histogram
        c = c[:len(h)]
        offset = (xmax - xmin) / (2 * bins)
        c += offset / 2
        grid = gd.make_grid()

        if mpaxes == None:
            plt.figure()
            ax = plt.axes()
        else:
            ax = mpaxes
        ax.plot(c + offset, h, linewidth=2)

        for k in range(self.k):
            ax.plot(grid, density[:, k], linewidth=2)
        ax.set_title('Fit of the density with a mixture of Gaussians',
                     fontsize=12)

        legend = ['data']
        for k in range(self.k):
            legend.append('component %d' % (k + 1))
        l = ax.legend(tuple(legend))
        for t in l.get_texts():
            t.set_fontsize(12)
        ax.set_xticklabels(ax.get_xticks(), fontsize=12)
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)

    def show(self, x, gd, density=None, axes=None):
        """
        Function to plot a GMM, still in progress
        Currently, works only in 1D and 2D

        Parameters
        ----------
        x: array of shape(n_samples, dim)
           the data under study
        gd: GridDescriptor instance
        density: array os shape(prod(gd.n_bins))
                 density of the model one the discrete grid implied by gd
                 by default, this is recomputed
        """
        import matplotlib.pyplot as plt

        # recompute the density if necessary
        if density is None:
            density = self.mixture_likelihood(gd, x)

        if axes is None:
            axes = plt.figure()

        if gd.dim == 1:
            from ..statistics.empirical_pvalue import \
                smoothed_histogram_from_samples
            h, c = smoothed_histogram_from_samples(x, normalized=True)
            offset = (c.max() - c.min()) / (2 * c.size)
            grid = gd.make_grid()

            h /= h.sum()
            h /= (2 * offset)
            plt.plot(c[: -1] + offset, h)
            plt.plot(grid, density)

        if gd.dim == 2:
            plt.figure()
            xm, xM, ym, yM = gd.lim[0:3]
            gd0 = gd.n_bins[0]
            Pdens = np.reshape(density, (gd0, np.size(density) / gd0))
            axes.imshow(Pdens.T, None, None, None, 'nearest',
                      1.0, None, None, 'lower', [xm, xM, ym, yM])
            axes.plot(x[:, 0], x[:, 1], '.k')
            axes.axis([xm, xM, ym, yM])
        return axes
