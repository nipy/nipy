"""
Implementation of Von-Mises-Fisher Mixture models,
i.e. the equaivalent of mixture of Gaussian on the sphere.

Author: Bertrand Thirion, 2010-2011
"""
import numpy as np


class VonMisesMixture(object):
    """
    Model for Von Mises mixture distribution with fixed variance
    on a two-dimensional sphere
    """

    def __init__(self, k, precision, means=None, weights=None,
                 null_class=False):
        """ Initialize Von Mises mixture

        Parameters
        ----------
        k: int,
           number of components
        precision: float,
                   the fixed precision parameter
        means: array of shape(self.k, 3), optional
               input component centers
        weights: array of shape(self.k), optional
                 input components weights
        null_class: bool, optional
                    Inclusion of a null class within the model
                    (related to k=0)

        fixme
        -----
        consistency checks
        """
        self.k = k
        self.dim = 2
        self.em_dim = 3
        self.means = means
        self.precision = precision
        self.weights = weights
        self.null_class = null_class

    def log_density_per_component(self, x):
        """Compute the per-component density of the data

        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        Returns
        -------
        like: array of shape(n, self.k), with non-neagtive values
              the density
        """
        n = x.shape[0]
        constant = self.precision / (2 * np.pi * (1 - np.exp( - \
                    2 * self.precision)))
        loglike = np.log(constant) + \
            (np.dot(x, self.means.T) - 1) * self.precision
        if self.null_class:
            loglike = np.hstack((np.log(1. / (4 * np.pi)) * np.ones((n, 1)),
                                 loglike))
        return loglike

    def density_per_component(self, x):
        """
        Compute the per-component density of the data

        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        Returns
        -------
        like: array of shape(n, self.k), with non-neagtive values
              the density
        """
        return np.exp(self.log_density_per_component(x))

    def weighted_density(self, x):
        """ Return weighted density

        Parameters
        ----------
        x: array shape(n,3)
           should be on the unit sphere

        Returns
        -------
        like: array
            of shape(n, self.k)
        """
        return(self.density_per_component(x) * self.weights)

    def log_weighted_density(self, x):
        """ Return log weighted density

        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        Returns
        -------
        log_like: array of shape(n, self.k)
        """
        return(self.log_density_per_component(x) + np.log(self.weights))

    def mixture_density(self, x):
        """ Return mixture density

        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        Returns
        -------
        like: array of shape(n)
        """
        wl = self.weighted_density(x)
        return np.sum(wl, 1)

    def responsibilities(self, x):
        """ Return responsibilities

        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere

        Returns
        -------
        resp: array of shape(n, self.k)
        """
        lwl = self.log_weighted_density(x)
        wl = np.exp(lwl.T - lwl.mean(1)).T
        swl = np.sum(wl, 1)
        resp = (wl.T / swl).T
        return resp

    def estimate_weights(self, z):
        """ Calculate and set weights from `z`

        Parameters
        ----------
        z: array of shape(self.k)
        """
        self.weights = np.sum(z, 0) / z.sum()

    def estimate_means(self, x, z):
        """ Calculate and set means from `x` and `z`

        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere
        z: array of shape(self.k)
        """
        m = np.dot(z.T, x)
        self.means = (m.T / np.sqrt(np.sum(m ** 2, 1))).T

    def estimate(self, x, maxiter=100, miniter=1, bias=None):
        """ Return average log density across samples

        Parameters
        ----------
        x: array of shape (n,3)
           should be on the unit sphere
        maxiter : int, optional
            maximum number of iterations of the algorithms
        miniter : int, optional
            minimum number of iterations
        bias : array of shape(n), optional
            prior probability of being in a non-null class

        Returns
        -------
        ll : float
            average (across samples) log-density
        """
        # initialization with random positions and constant weights
        if self.weights is None:
            self.weights = np.ones(self.k) / self.k
            if self.null_class:
                self.weights = np.ones(self.k + 1) / (self.k + 1)

        if self.means is None:
            aux = np.arange(x.shape[0])
            np.random.shuffle(aux)
            self.means = x[aux[:self.k]]

        # EM algorithm
        assert not(np.isnan(self.means).any())
        pll = - np.inf
        for i in range(maxiter):
            ll = np.log(self.mixture_density(x)).mean()
            z = self.responsibilities(x)
            assert not(np.isnan(z).any())

            # bias z
            if bias is not None:
                z[:, 0] *= (1 - bias)
                z[:, 1:] = ((z[:, 1:].T) * bias).T
                z = (z.T / np.sum(z, 1)).T

            self.estimate_weights(z)
            if self.null_class:
                self.estimate_means(x, z[:, 1:])
            else:
                self.estimate_means(x, z)
            assert not(np.isnan(self.means).any())
            if (i > miniter) and (ll < pll + 1.e-6):
                break
            pll = ll
        return ll

    def show(self, x):
        """ Visualization utility

        Parameters
        ----------
        x: array fo shape(n,3)
           should be on the unit sphere
        """
        # label the data
        z = np.argmax(self.responsibilities(x), 1)
        import pylab
        import mpl_toolkits.mplot3d.axes3d as p3
        fig = pylab.figure()
        ax = p3.Axes3D(fig)
        colors = (['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * \
                      (1 + (1 + self.k) / 8))[:self.k + 1]
        if (self.null_class) and (z == 0).any():
            ax.plot3D(x[z == 0, 0], x[z == 0, 1], x[z == 0, 2], '.',
                      color=colors[0])
        for k in range(self.k):
            if self.null_class:
                if np.sum(z == (k + 1)) == 0:
                    continue
                uk = z == (k + 1)
                ax.plot3D(x[uk, 0], x[uk, 1], x[uk, 2], '.',
                          color=colors[k + 1])
                ax.plot3D([self.means[k, 0]], [self.means[k, 1]],
                      [self.means[k, 2]], 'o', color=colors[k + 1])
            else:
                if np.sum(z == k) == 0:
                    continue
                ax.plot3D(x[z == k, 0], x[z == k, 1], x[z == k, 2], '.',
                              color=colors[k])
                ax.plot3D([self.means[k, 0]], [self.means[k, 1]],
                      [self.means[k, 2]], 'o', color=colors[k])
        pylab.show()


def estimate_robust_vmm(k, precision, null_class, x, ninit=10, bias=None,
                        maxiter=100):
    """ Return the best von_mises mixture after severla initialization

    Parameters
    ----------
    k: int, number of classes
    precision: float, priori precision parameter
    null class: bool, optional,
                should a null class be included or not
    x: array fo shape(n,3)
       input data, should be on the unit sphere
    ninit: int, optional,
           number of iterations
    bias: array of shape(n), optional
          prior probability of being in a non-null class
    maxiter: int, optional,
             maximum number of iterations after each initialization
    """
    score = - np.inf
    for i in range(ninit):
        aux = VonMisesMixture(k, precision, null_class=null_class)
        ll = aux.estimate(x, bias=bias)
        if ll > score:
            best_model = aux
            score = ll
    return best_model


def select_vmm(krange, precision, null_class, x, ninit=10, bias=None,
               maxiter=100, verbose=0):
    """Return the best von_mises mixture after severla initialization

    Parameters
    ----------
    krange: list of ints,
            number of classes to consider
    precision:
    null class:
    x: array fo shape(n,3)
       should be on the unit sphere
    ninit: int, optional,
           number of iterations
    maxiter: int, optional,
    bias: array of shape(n),
          a prior probability of not being in the null class
    verbose: Bool, optional
    """
    score = - np.inf
    for k in krange:
        aux = estimate_robust_vmm(k, precision, null_class, x, ninit, bias,
                                  maxiter)
        ll = aux.estimate(x)
        if null_class:
            bic = ll - np.log(x.shape[0]) * k * 3 / x.shape[0]
        else:
            bic = ll - np.log(x.shape[0]) * (k * 3 - 1) / x.shape[0]
        if verbose:
            print k, bic
        if bic > score:
            best_model = aux
            score = bic
    return best_model


def select_vmm_cv(krange, precision, x, null_class, cv_index,
                  ninit=5, maxiter=100, bias=None, verbose=0):
    """Return the best von_mises mixture after severla initialization

    Parameters
    ----------
    krange: list of ints,
            number of classes to consider
    precision: float,
               precision parameter of the von-mises densities
    x: array fo shape(n, 3)
       should be on the unit sphere
    null class: bool, whether a null class should be included or not
    cv_index: set of indices for cross validation
    ninit: int, optional,
           number of iterations
    maxiter: int, optional,
    bias: array of shape (n), prior
    """
    score = - np.inf
    mll = []
    for k in krange:
        mll.append( - np.inf)
        for j in range(1):
            ll = np.zeros_like(cv_index).astype(np.float)
            for i in np.unique(cv_index):
                xl = x[cv_index != i]
                xt = x[cv_index == i]
                bias_l = None
                if bias is not None:
                    bias_l = bias[cv_index != i]
                aux = estimate_robust_vmm(k, precision, null_class, xl,
                                          ninit=ninit, bias=bias_l,
                                          maxiter=maxiter)
                if bias is None:
                    ll[cv_index == i] = np.log(aux.mixture_density(xt))
                else:
                    bias_t = bias[cv_index == i]
                    lwd = aux.weighted_density(xt)
                    ll[cv_index == i] = np.log(lwd[:, 0] * (1 - bias_t) +  \
                        lwd[:, 1:].sum(1) * bias_t)
            if ll.mean() > mll[-1]:
                mll[-1] = ll.mean()

        aux = estimate_robust_vmm(k, precision, null_class, x,
                                  ninit, bias=bias, maxiter=maxiter)

        if verbose:
            print k, mll[ - 1]
        if mll[ - 1] > score:
            best_model = aux
            score = mll[ - 1]

    return best_model


def sphere_density(npoints):
    """Return the points and area of a npoints**2 points sampled on a sphere

    Returns
    -------
    s : array of shape(npoints ** 2, 3)
    area: array of shape(npoints)
    """
    u = np.linspace(0, 2 * np.pi, npoints + 1)[:npoints]
    v = np.linspace(0, np.pi, npoints + 1)[:npoints]
    s = np.vstack((np.ravel(np.outer(np.cos(u), np.sin(v))),
                np.ravel(np.outer(np.sin(u), np.sin(v))),
                   np.ravel(np.outer(np.ones(np.size(u)), np.cos(v))))).T
    area = np.abs(np.ravel(np.outer(np.ones(np.size(u)), np.sin(v)))) * \
           np.pi ** 2 * 2 * 1. / (npoints ** 2)
    return s, area


def example_noisy():
    x1 = [0.6, 0.48, 0.64]
    x2 = [-0.8, 0.48, 0.36]
    x3 = [0.48, 0.64, -0.6]
    x = np.random.randn(200, 3) * .1
    x[:30] += x1
    x[40:150] += x2
    x[150:] += x3
    x = (x.T / np.sqrt(np.sum(x ** 2, 1))).T

    precision = 100.
    vmm = select_vmm(range(2, 7), precision, True, x)
    vmm.show(x)

    # check that it sums to 1
    s, area = sphere_density(100)
    print (vmm.mixture_density(s) * area).sum()


def example_cv_nonoise():
    x1 = [0.6, 0.48, 0.64]
    x2 = [-0.8, 0.48, 0.36]
    x3 = [0.48, 0.64, -0.6]
    x = np.random.randn(30, 3) * .1
    x[0::3] += x1
    x[1::3] += x2
    x[2::3] += x3
    x = (x.T / np.sqrt(np.sum(x ** 2, 1))).T

    precision = 50.
    sub = np.repeat(np.arange(10), 3)
    vmm = select_vmm_cv(range(1, 8), precision, x, cv_index=sub,
                        null_class=False, ninit=20)
    vmm.show(x)

    # check that it sums to 1
    s, area = sphere_density(100)
    return vmm
