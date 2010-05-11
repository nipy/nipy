"""
Test the Infinite GMM.

Author : Bertrand Thirion, 2010 
"""

#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

import numpy as np
import numpy.random as nr
from nipy.neurospin.clustering.imm import IMM


def test_imm_loglike_1D():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood
    """
    n = 100
    dim = 1
    alpha = .5
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100)

    # sampling
    like =  igmm.sample(x, niter=300)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.25*dim

def test_imm_loglike_1D_k10():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood

    Here k-fold cross validation is used(k=10)
    """
    n = 100
    dim = 1
    alpha = .5
    k = 10
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100, k=k)

    # sampling
    like =  igmm.sample(x, niter=300, k=k)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.25*dim


def test_imm_loglike_2D_fast():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood

    fast version
    """
    n = 100
    dim = 2
    alpha = .5
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100, init=True)

    # sampling
    like =  igmm.sample(x, niter=300)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.25*dim

def test_imm_loglike_2D():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood

    slow (cross-validated) but accurate.
    """
    n = 100
    dim = 2
    alpha = .5
    k = 10
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100, init=True, k=k)

    # sampling
    like =  igmm.sample(x, niter=300, k=k)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.25*dim


def test_imm_loglike_2D_a0_1():
    """
    Chek that the log-likelihood of the data under the
    infinite gaussian mixture model
    is close to the theortical data likelihood

    the difference is that now alpha=.1
    """
    n = 100
    dim = 2
    alpha = .1
    x = np.random.randn(n, dim)
    igmm = IMM(alpha, dim)
    igmm.set_priors(x)

    # warming
    igmm.sample(x, niter=100, init=True)

    # sampling
    like =  igmm.sample(x, niter=300)
    theoretical_ll = -dim*.5*(1+np.log(2*np.pi))
    empirical_ll = np.log(like).mean()
    print theoretical_ll, empirical_ll
    assert np.absolute(theoretical_ll-empirical_ll)<0.2*dim



if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


