"""
Infinite mixture model : A generalization of Bayesian mixture models
with an unspecified number of classes
"""
import numpy as np
from bgmm import BGMM, detsh
from scipy.special import gammaln


def co_labelling(z, kmax=None, kmin=None):
    """
    return a sparse co-labelling matrix given the label vector z

    Parameters
    ----------
    z: array of shape(n_samples),
       the input labels
    kmax: int, optional,
          considers only the labels in the range [0, kmax[

    Returns
    -------
    colabel: a sparse coo_matrix,
             yields the co labelling of the data
             i.e. c[i,j]= 1 if z[i]==z[j], 0 otherwise
    """
    from scipy.sparse import coo_matrix
    n = z.size
    colabel = coo_matrix((n, n))

    if kmax == None:
        kmax = z.max() + 1

    if kmin == None:
        kmin = z.min() - 1

    for  k in np.unique(z):
        if (k < kmax) & (k > kmin):
            i = np.array(np.nonzero(z == k))
            row = np.repeat(i, i.size)
            col = np.ravel(np.tile(i, i.size))
            data = np.ones((i.size) ** 2)
            colabel = colabel + coo_matrix((data, (row, col)), shape=(n, n))
    return colabel


class IMM(BGMM):
    """
    The class implements Infinite Gaussian Mixture model
    or Dirichlet Proces Mixture Model.
    This simply a generalization of Bayesian Gaussian Mixture Models
    with an unknown number of classes.
    """

    def __init__(self, alpha=.5, dim=1):
        """
        Parameters
        ----------
        alpha: float, optional,
               the parameter for cluster creation
        dim: int, optional,
             the dimension of the the data

        Note: use the function set_priors() to set adapted priors
        """
        self.dim = dim
        self.alpha = alpha
        self.k = 0
        self.prec_type = 'full'

        # initialize weights
        self.weights = [1]

    def set_priors(self, x):
        """ Set the priors in order of having them weakly uninformative
        this is from  Fraley and raftery;
        Journal of Classification 24:155-181 (2007)

        Parameters
        ----------
        x, array of shape (n_samples,self.dim)
           the data used in the estimation process
        """
        # a few parameters
        small = 0.01
        elshape = (1, self.dim, self.dim)
        mx = np.reshape(x.mean(0), (1, self.dim))
        dx = x - mx
        vx = np.maximum(1.e-15, np.dot(dx.T, dx) / x.shape[0])
        px = np.reshape(np.diag(1.0 / np.diag(vx)), elshape)

        # set the priors
        self._prior_means = mx
        self.prior_means = mx
        self.prior_weights = self.alpha
        self._prior_scale = px
        self.prior_scale = px
        self._prior_dof = self.dim + 2
        self.prior_dof = [self._prior_dof]
        self._prior_shrinkage = small
        self.prior_shrinkage = [self._prior_shrinkage]

        # cache some pre-computations
        self._dets_ = detsh(px[0])
        self._dets = [self._dets_]
        self._inv_prior_scale_ = np.reshape(np.linalg.inv(px[0]), elshape)
        self.prior_dens = None

    def set_constant_densities(self, prior_dens=None):
        """Set the null and prior densities as constant
        (assuming a compact domain)

        Parameters
        ----------
        prior_dens: float, optional
                    constant for the prior density
        """
        self.prior_dens = prior_dens

    def sample(self, x, niter=1, sampling_points=None, init=False,
               kfold=None, verbose=0):
        """sample the indicator and parameters

        Parameters
        ----------
        x: array of shape (n_samples, self.dim)
           the data used in the estimation process
        niter: int,
               the number of iterations to perform
        sampling_points: array of shape(nbpoints, self.dim), optional
                         points where the likelihood will be sampled
                         this defaults to x
        kfold: int or array, optional,
               parameter of cross-validation control
               by default, no cross-validation is used
               the procedure is faster but less accurate
        verbose=0: verbosity mode

        Returns
        -------
        likelihood: array of shape(nbpoints)
                    total likelihood of the model
        """
        self.check_x(x)
        if sampling_points == None:
            average_like = np.zeros(x.shape[0])
        else:
            average_like = np.zeros(sampling_points.shape[0])
            splike = self.likelihood_under_the_prior(sampling_points)

        plike = self.likelihood_under_the_prior(x)

        if init:
            self.k = 1
            z = np.zeros(x.shape[0])
            self.update(x, z)

        like = self.likelihood(x, plike)
        z = self.sample_indicator(like)

        for i in range(niter):
            if  kfold == None:
                like = self.simple_update(x, z, plike)
            else:
                like = self.cross_validated_update(x, z, plike, kfold)

            if sampling_points == None:
                average_like += like
            else:
                average_like += np.sum(
                    self.likelihood(sampling_points, splike), 1)

        average_like /= niter
        return average_like

    def simple_update(self, x, z, plike):
        """
         This is a step in the sampling procedure
        that uses internal corss_validation

        Parameters
        ----------
        x: array of shape(n_samples, dim),
           the input data
        z: array of shape(n_samples),
           the associated membership variables
        plike: array of shape(n_samples),
               the likelihood under the prior

        Returns
        -------
        like: array od shape(n_samples),
              the likelihood of the data
        """
        like = self.likelihood(x, plike)
        # standard + likelihood under the prior
        # like has shape (x.shape[0], self.k+1)

        z = self.sample_indicator(like)
        # almost standard, but many new components can be created

        self.reduce(z)
        self.update(x, z)
        return like.sum(1)

    def cross_validated_update(self, x, z, plike, kfold=10):
        """
        This is a step in the sampling procedure
        that uses internal corss_validation

        Parameters
        ----------
        x: array of shape(n_samples, dim),
           the input data
        z: array of shape(n_samples),
           the associated membership variables
        plike: array of shape(n_samples),
               the likelihood under the prior
        kfold: int, or array of shape(n_samples), optional,
               folds in the cross-validation loop

        Returns
        -------
        like: array od shape(n_samples),
              the (cross-validated) likelihood of the data
        """
        n_samples = x.shape[0]
        slike = np.zeros(n_samples)

        if np.isscalar(kfold):
            aux = np.argsort(np.random.rand(n_samples))
            idx = - np.ones(n_samples).astype(np.int)
            j = np.ceil(n_samples / kfold)
            kmax = kfold
            for k in range(kmax):
                idx[aux[k * j:min(n_samples, j * (k + 1))]] = k
        else:
            if np.array(kfold).size != n_samples:
                raise ValueError('kfold and x do not have the same size')
            uk = np.unique(kfold)
            np.random.shuffle(uk)
            idx = np.zeros(n_samples).astype(np.int)
            for i, k in enumerate(uk):
                idx += (i * (kfold == k))
            kmax = uk.max() + 1

        for k in range(kmax):
            test = np.zeros(n_samples).astype('bool')
            test[idx == k] = 1
            train = np.logical_not(test)

            # remove a fraction of the data
            # and re-estimate the clusters
            z[train] = self.reduce(z[train])
            self.update(x[train], z[train])

            # draw the membership for the left-out datas
            alike = self.likelihood(x[test], plike[test])
            slike[test] = alike.sum(1)
            # standard + likelihood under the prior
            # like has shape (x.shape[0], self.k+1)

            z[test] = self.sample_indicator(alike)
            # almost standard, but many new components can be created

        return slike

    def reduce(self, z):
        """Reduce the assignments by removing empty clusters and update self.k

        Parameters
        ----------
        z: array of shape(n),
           a vector of membership variables changed in place

        Returns
        -------
        z: the remapped values
        """
        uz = np.unique(z[z > - 1])
        for i, k in enumerate(uz):
            z[z == k] = i
        self.k = z.max() + 1
        return z

    def update(self, x, z):
        """ Update function (draw a sample of the IMM parameters)

        Parameters
        ----------
        x array of shape (n_samples,self.dim)
          the data used in the estimation process
        z array of shape (n_samples), type = np.int
          the corresponding classification
        """
        # re-dimension the priors in order to match self.k
        self.prior_means = np.repeat(self._prior_means, self.k, 0)
        self.prior_dof = self._prior_dof * np.ones(self.k)
        self.prior_shrinkage = self._prior_shrinkage * np.ones(self.k)
        self._dets = self._dets_ * np.ones(self.k)
        self._inv_prior_scale = np.repeat(self._inv_prior_scale_, self.k, 0)

        # initialize some variables
        self.means = np.zeros((self.k, self.dim))
        self.precisions = np.zeros((self.k, self.dim, self.dim))

        # proceed with the update
        BGMM.update(self, x, z)

    def update_weights(self, z):
        """
        Given the allocation vector z, resmaple the weights parameter

        Parameters
        ----------
        z array of shape (n_samples), type = np.int
          the allocation variable
        """
        pop = np.hstack((self.pop(z), 0))
        self.weights = pop + self.prior_weights
        self.weights /= self.weights.sum()

    def sample_indicator(self, like):
        """ Sample the indicator from the likelihood

        Parameters
        ----------
        like: array of shape (nbitem,self.k)
           component-wise likelihood

        Returns
        -------
        z: array of shape(nbitem): a draw of the membership variable

        Notes
        -----
        The behaviour is different from standard bgmm in that z can take
        arbitrary values
        """
        z = BGMM.sample_indicator(self, like)
        z[z == self.k] = self.k + np.arange(np.sum(z == self.k))
        return z

    def likelihood_under_the_prior(self, x):
        """ Computes the likelihood of x under the prior

        Parameters
        ----------
        x, array of shape (self.n_samples,self.dim)

        returns
        -------
        w, the likelihood of x under the prior model (unweighted)
        """
        if self.prior_dens is not None:
            return self.prior_dens * np.ones(x.shape[0])

        a = self._prior_dof
        tau = self._prior_shrinkage
        tau /= (1 + tau)
        m = self._prior_means
        b = self._prior_scale
        ib = np.linalg.inv(b[0])
        ldb = np.log(detsh(b[0]))

        scalar_w = np.log(tau / np.pi) * self.dim
        scalar_w += 2 * gammaln((a + 1) / 2)
        scalar_w -= 2 * gammaln((a - self.dim) / 2)
        scalar_w -= ldb * a
        w = scalar_w * np.ones(x.shape[0])

        for i in range(x.shape[0]):
            w[i] -= (a + 1) * np.log(detsh(ib + tau * (m - x[i:i + 1]) *
                                           (m - x[i:i + 1]).T))

        w /= 2
        return np.exp(w)

    def likelihood(self, x, plike=None):
        """
        return the likelihood of the model for the data x
        the values are weighted by the components weights

        Parameters
        ----------
        x: array of shape (n_samples, self.dim),
           the data used in the estimation process
        plike: array os shape (n_samples), optional,x
               the desnity of each point under the prior

        Returns
        -------
        like, array of shape(nbitem,self.k)
        component-wise likelihood
        """
        if plike == None:
            plike = self.likelihood_under_the_prior(x)

        plike = np.reshape(plike, (x.shape[0], 1))
        if self.k > 0:
            like = self.unweighted_likelihood(x)
            like = np.hstack((like, plike))
        else:
            like = plike
        like *= self.weights
        return like


class MixedIMM(IMM):
    """
    Particular IMM with an additional null class.
    The data is supplied together
    with a sample-related probability of being under the null.
    """

    def __init__(self, alpha=.5, dim=1):
        """
        Parameters
        ----------
        alpha: float, optional,
               the parameter for cluster creation
        dim: int, optional,
             the dimension of the the data

        Note: use the function set_priors() to set adapted priors
        """
        IMM.__init__(self, alpha, dim)

    def set_constant_densities(self, null_dens=None, prior_dens=None):
        """
        Set the null and prior densities as constant
        (over a  supposedly compact domain)

        Parameters
        ----------
        null_dens: float, optional
                   constant for the null density
        prior_dens: float, optional
                    constant for the prior density
        """
        self.null_dens = null_dens
        self.prior_dens = prior_dens

    def sample(self, x, null_class_proba, niter=1, sampling_points=None,
               init=False, kfold=None, co_clustering=False, verbose=0):
        """
        sample the indicator and parameters

        Parameters
        ----------
        x: array of shape (n_samples, self.dim),
           the data used in the estimation process
        null_class_proba: array of shape(n_samples),
                          the probability to be under the null
        niter: int,
               the number of iterations to perform
        sampling_points: array of shape(nbpoints, self.dim), optional
                         points where the likelihood will be sampled
                         this defaults to x
        kfold: int, optional,
               parameter of cross-validation control
               by default, no cross-validation is used
               the procedure is faster but less accurate
        co_clustering: bool, optional
                       if True,
                       return a model of data co-labelling across iterations
        verbose=0: verbosity mode

        Returns
        -------
        likelihood: array of shape(nbpoints)
                    total likelihood of the model
        pproba: array of shape(n_samples),
                the posterior of being in the null
                (the posterior of null_class_proba)
        coclust: only if co_clustering==True,
                 sparse_matrix of shape (n_samples, n_samples),
                 frequency of co-labelling of each sample pairs
                 across iterations
        """
        self.check_x(x)
        pproba = np.zeros(x.shape[0])

        if sampling_points == None:
            average_like = np.zeros(x.shape[0])
        else:
            average_like = np.zeros(sampling_points.shape[0])
            splike = self.likelihood_under_the_prior(sampling_points)

        plike = self.likelihood_under_the_prior(x)

        if init:
            self.k = 1
            z = np.zeros(x.shape[0])
            self.update(x, z)

        like = self.likelihood(x, plike)
        z = self.sample_indicator(like, null_class_proba)

        if co_clustering:
            from scipy.sparse import coo_matrix
            coclust = coo_matrix((x.shape[0], x.shape[0]))

        for i in range(niter):
            if  kfold == None:
                like = self.simple_update(x, z, plike, null_class_proba)
            else:
                like, z = self.cross_validated_update(x, z, plike,
                                                      null_class_proba, kfold)

            llike = self.likelihood(x, plike)
            z = self.sample_indicator(llike, null_class_proba)
            pproba += (z == - 1)

            if co_clustering:
                coclust = coclust + co_labelling(z, self.k, -1)

            if sampling_points == None:
                average_like += like
            else:
                average_like += np.sum(
                    self.likelihood(sampling_points, splike), 1)

        average_like /= niter
        pproba /= niter
        if co_clustering:
            coclust /= niter
            return average_like, pproba, coclust
        return average_like, pproba

    def simple_update(self, x, z, plike, null_class_proba):
        """ One step in the sampling procedure (one data sweep)

        Parameters
        ----------
        x: array of shape(n_samples, dim),
           the input data
        z: array of shape(n_samples),
           the associated membership variables
        plike: array of shape(n_samples),
               the likelihood under the prior
        null_class_proba: array of shape(n_samples),
                          prior probability to be under the null

        Returns
        -------
        like: array od shape(n_samples),
              the likelihood of the data under the H1 hypothesis
        """
        like = self.likelihood(x, plike)
        # standard + likelihood under the prior
        # like has shape (x.shape[0], self.k+1)

        z = self.sample_indicator(like, null_class_proba)
        # almost standard, but many new components can be created

        self.reduce(z)
        self.update(x, z)
        return like.sum(1)

    def cross_validated_update(self, x, z, plike, null_class_proba, kfold=10):
        """
        This is a step in the sampling procedure
        that uses internal corss_validation

        Parameters
        ----------
        x: array of shape(n_samples, dim),
           the input data
        z: array of shape(n_samples),
           the associated membership variables
        plike: array of shape(n_samples),
               the likelihood under the prior
        kfold: int, optional, or array
               number of folds in cross-validation loop
               or set of indexes for the cross-validation procedure
        null_class_proba: array of shape(n_samples),
                          prior probability to be under the null

        Returns
        -------
        like: array od shape(n_samples),
              the (cross-validated) likelihood of the data
        z: array of shape(n_samples),
              the associated membership variables

        Notes
        -----
        When kfold is an array, there is an internal reshuffling to randomize
        the order of updates
        """
        n_samples = x.shape[0]
        slike = np.zeros(n_samples)

        if np.isscalar(kfold):
            aux = np.argsort(np.random.rand(n_samples))
            idx = - np.ones(n_samples).astype(np.int)
            j = np.ceil(n_samples / kfold)
            kmax = kfold
            for k in range(kmax):
                idx[aux[k * j:min(n_samples, j * (k + 1))]] = k
        else:
            if np.array(kfold).size != n_samples:
                raise ValueError('kfold and x do not have the same size')
            uk = np.unique(kfold)
            np.random.shuffle(uk)
            idx = np.zeros(n_samples).astype(np.int)
            for i, k in enumerate(uk):
                idx += (i * (kfold == k))
            kmax = uk.max() + 1

        for k in range(kmax):
            # split at iteration k
            test = np.zeros(n_samples).astype('bool')
            test[idx == k] = 1
            train = np.logical_not(test)

            # remove a fraction of the data
            # and re-estimate the clusters
            z[train] = self.reduce(z[train])
            self.update(x[train], z[train])

            # draw the membership for the left-out data
            alike = self.likelihood(x[test], plike[test])
            slike[test] = alike.sum(1)
            # standard + likelihood under the prior
            # like has shape (x.shape[0], self.k+1)

            z[test] = self.sample_indicator(alike, null_class_proba[test])
            # almost standard, but many new components can be created

        return slike, z

    def sample_indicator(self, like, null_class_proba):
        """
        sample the indicator from the likelihood

        Parameters
        ----------
        like: array of shape (nbitem,self.k)
           component-wise likelihood
        null_class_proba: array of shape(n_samples),
                          prior probability to be under the null

        Returns
        -------
        z: array of shape(nbitem): a draw of the membership variable

        Notes
        -----
        Here z=-1 encodes for the null class
        """
        n = like.shape[0]
        conditional_like_1 = ((1 - null_class_proba) * like.T).T
        conditional_like_0 = np.reshape(null_class_proba *
                                        self.null_dens, (n, 1))
        conditional_like = np.hstack((conditional_like_0, conditional_like_1))
        z = BGMM.sample_indicator(self, conditional_like) - 1
        z[z == self.k] = self.k + np.arange(np.sum(z == self.k))
        return z


def main():
    """ Illustrative example of the behaviour of imm
    """
    n = 100
    dim = 2
    alpha = .5
    aff = np.random.randn(dim, dim)
    x = np.dot(np.random.randn(n, dim), aff)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100, kfold=10)
    print 'number of components: ', igmm.k

    #
    print 'number of components: ', igmm.k

    if dim < 3:
        from gmm import plot2D
        plot2D(x, igmm, verbose=1)
    return igmm


if __name__ == '__main__':
    main()
