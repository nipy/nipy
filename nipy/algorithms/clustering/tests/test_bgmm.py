"""
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
Test the Bayesian GMM.

fixme : some of these tests take too much time at the moment
to be real unit tests

Author : Bertrand Thirion, 2009
"""

import numpy as np
import numpy.random as nr

from ..bgmm import BGMM, VBGMM, dirichlet_eval, multinomial, dkl_gaussian 

from nose.tools import assert_true

def test_dirichlet_eval():
    # check that the Dirichlet evaluation function sums to one on a simple
    # example
    alpha = np.array([0.5, 0.5])
    sd = 0
    for i in range(10000):
        e = i * 0.0001 + 0.00005
        sd += dirichlet_eval(np.array([e, 1 - e]), alpha)
    assert_true(np.absolute(sd.sum() * 0.0001 - 1) < 0.01)


def test_multinomial():
    """
    test of the generate_multinomial function:
    check that is sums to 1 in a simple case
    """
    n_samples = 100000
    n_classes = 5
    aux = np.reshape(np.random.rand(n_classes), (1, n_classes))
    aux /= aux.sum()
    likelihood = np.repeat(aux, n_samples, 0)
    z = multinomial(likelihood)
    res = np.array([np.sum(z == k) for k in range(n_classes)])
    res = res * 1.0 / n_samples
    assert_true(np.sum((aux-res) ** 2) < 1.e-4)


def test_dkln1():
    dim = 3
    m1 = np.zeros(dim)
    P1 = np.eye(dim)
    m2 = m1
    P2 = P1
    assert_true(dkl_gaussian(m1, P1, m2, P2) == 0)


def test_dkln2():
    dim, offset = 3, 4.
    m1 = np.zeros(dim)
    P1 = np.eye(dim)
    m2 = offset * np.ones(dim)
    P2 = np.eye(dim)
    assert_true(dkl_gaussian(m1, P1, m2, P2) == .5 * dim * offset ** 2)


def test_dkln3():
    dim, scale = 3, 4
    m1, m2 = np.zeros(dim), np.zeros(dim)
    P1, P2 = np.eye(dim), scale * np.eye(dim)
    test1 = .5 * (dim * np.log(scale) + dim * (1. / scale - 1))
    test2 = .5 * (-dim * np.log(scale) + dim * (scale - 1))
    assert_true(dkl_gaussian(m1, P1, m2, P2) == test2)


def test_bgmm_gibbs():
    # Perform the estimation of a gmm using Gibbs sampling
    n_samples, k, dim, niter, offset = 100, 2, 2, 1000, 2.
    x = nr.randn(n_samples,dim)
    x[:30] += offset

    b = BGMM(k,dim)
    b.guess_priors(x)
    b.initialize(x)
    b.sample(x, 1)
    w, cent, prec, pz = b.sample(x, niter, mem=1)
    b.plugin(cent, prec, w)
    z = pz[:, 0]

    # fixme : find a less trivial test
    assert_true(z.max() + 1 == b.k)


def test_gmm_bf(kmax=4, seed=1):
    """ Perform a model selection procedure on a gmm
    with Bayes factor estimations

    Parameters
    ----------
    kmax : range of values that are tested
    seed=False:  int, optionnal
        If seed is not False, the random number generator is initialized
        at a certain value

    fixme : this one often fails. I don't really see why
    """
    n_samples, dim, niter = 30, 2, 1000

    if seed:
        nr = np.random.RandomState([seed])
    else:
        import numpy.random as nr

    x = nr.randn(n_samples, dim)

    bbf = -np.inf
    for k in range(1, kmax):
        b = BGMM(k, dim)
        b.guess_priors(x)
        b.initialize(x)
        b.sample(x, 100)
        w, cent, prec, pz = b.sample(x, niter=niter, mem=1)
        bplugin =  BGMM(k, dim, cent, prec, w)
        bplugin.guess_priors(x)
        bfk = bplugin.bayes_factor(x, pz.astype(np.int))
        if bfk > bbf:
            bestk = k
            bbf = bfk
    assert_true(bestk < 3)


def test_vbgmm():
    """perform the estimation of a variational gmm
    """
    n_samples, dim, offset, k = 100, 2, 2, 2
    x = nr.randn(n_samples, dim)
    x[:30] += offset
    b = VBGMM(k,dim)
    b.guess_priors(x)
    b.initialize(x)
    b.estimate(x)
    z = b.map_label(x)

    # fixme : find a less trivial test
    assert_true(z.max() + 1 == b.k)


def test_vbgmm_select(kmax=6):
    """ perform the estimation of a variational gmm + model selection
    """
    nr.seed([0])
    n_samples, dim, offset=100, 3, 2
    x = nr.randn(n_samples, dim)
    x[:30] += offset
    be = - np.inf
    for  k in range(1, kmax):
        b = VBGMM(k, dim)
        b.guess_priors(x)
        b.initialize(x)
        b.estimate(x)
        ek = b.evidence(x)
        if ek > be:
            be = ek
            bestk = k
    assert_true(bestk < 3)


def test_evidence(k=1):
    """
    Compare the evidence estimated by Chib's method
    with the variational evidence (free energy)
    fixme : this one really takes time
    """
    np.random.seed(0)
    n_samples, dim, offset = 50, 2, 3
    x = nr.randn(n_samples, dim)
    x[:15] += offset

    b = VBGMM(k, dim)
    b.guess_priors(x)
    b.initialize(x)
    b.estimate(x)
    vbe = b.evidence(x)

    niter = 1000
    b = BGMM(k, dim)
    b.guess_priors(x)
    b.initialize(x)
    b.sample(x, 100)
    w, cent, prec, pz = b.sample(x, niter=niter, mem=1)
    bplugin =  BGMM(k, dim, cent, prec, w)
    bplugin.guess_priors(x)
    bfchib = bplugin.bayes_factor(x, pz.astype(np.int), 1)

    assert_true(bfchib > vbe)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
