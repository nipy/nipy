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


def test_dirichlet_eval():
    """
    check that the Dirichlet evaluation functiona sums to one
    on a simple example
    """
    alpha = np.array([0.5,0.5])
    sd = 0
    for i in range(10000):
        e = i*0.0001 + 0.00005
        sd += dirichlet_eval(np.array([e,1-e]),alpha)
    print sd.sum()
    assert(np.absolute(sd.sum()*0.0001-1)<0.01)


def test_multinomial():
    """
    test of the generate_multinomial function:
    check that is sums to 1 in a simple case
    """
    nsamples = 100000
    nclasses = 5
    aux = np.reshape(np.random.rand(nclasses),(1,nclasses))
    aux /= aux.sum()
    Likelihood = np.repeat(aux,nsamples,0)
    z = multinomial(Likelihood)
    res = np.array([np.sum(z==k) for k in range(nclasses)])
    res = res*1.0/nsamples
    assert np.sum((aux-res)**2)<1.e-4


def test_dkln1():
    m1 = np.zeros(3)
    P1 = np.eye(3)
    m2 = np.zeros(3)
    P2 = np.eye(3)
    assert dkl_gaussian(m1,P1,m2,P2)== 0


def test_dkln2():
    dim = 3
    offset = 4
    m1 = np.zeros(dim)
    P1 = np.eye(dim)
    m2 = offset*np.ones(dim)
    P2 = np.eye(dim)
    assert dkl_gaussian(m1,P1,m2,P2)==.5*dim*offset**2


def test_dkln3():
    dim = 3
    scale = 4
    m1 = np.zeros(dim)
    P1 = np.eye(dim)
    m2 = np.zeros(dim)
    P2 = scale*np.eye(dim)
    test1 = .5*(dim*np.log(scale)+dim*(1./scale-1))
    test2 = .5*(-dim*np.log(scale)+dim*(scale-1)) 
    print dkl_gaussian(m1,P1,m2,P2),test1,test2
    assert dkl_gaussian(m1,P1,m2,P2)==test2
  

def test_bgmm_gibbs(verbose=0):
    """
    perform the estimation of a gmm using Gibbs sampling
    """
    n=100
    k=2
    dim=2
    niter = 1000
    x = nr.randn(n,dim)
    x[:30] += 2
    
    b = BGMM(k,dim)
    b.guess_priors(x)
    b.initialize(x)
    b.sample(x,1)
    w,cent,prec,pz = b.sample(x, niter, mem=1)
    b.plugin(cent,prec,w)
    z = pz[:,0]
    
    # fixme : find a less trivial test
    assert(z.max()+1==b.k)


def test_gmm_bf(kmax=4, seed=1, verbose=1):
    """
    perform a model selection procedure on a  gmm
    with Bayes factor estimations

    Parameters
    ----------
    kmax : range of values that are tested
    seed=False:  int, optionnal
        If seed is not False, the random number generator is initialized
        at a certain value
        
    fixme : this one often fails. I don't really see why
    """
    n=30
    dim=2
    if seed:
        nr = np.random.RandomState([seed])
    else:
        import numpy.random as nr

    x = nr.randn(n,dim)
    niter = 1000

    bbf = -np.infty
    for k in range(1, kmax):
        b = BGMM(k, dim)
        b.guess_priors(x)
        b.initialize(x)
        b.sample(x, 100)
        w, cent, prec, pz = b.sample(x, niter=niter, mem=1)
        bplugin =  BGMM(k, dim, cent, prec, w)
        bplugin.guess_priors(x)
        bfk = bplugin.bayes_factor(x, pz.astype(np.int))
        if verbose:
            print k, bfk
        if bfk>bbf:
            bestk = k
            bbf = bfk
    assert(bestk<3)
    

def test_vbgmm(verbose=0):
    """
    perform the estimation of a gmm
    """
    n=100
    dim=2
    x = nr.randn(n,dim)
    x[:30] += 2
    k=2
    b = VBGMM(k,dim)
    b.guess_priors(x)
    b.initialize(x)
    b.estimate(x,verbose=verbose)
    z = b.map_label(x)
    
    # fixme : find a less trivial test
    assert(z.max()+1==b.k)


def test_vbgmm_select(kmax = 6,verbose=0):
    """
    perform the estimation of a gmm
    """
    n=100
    dim=3
    x = nr.randn(n,dim)
    x[:30] += 2
    be = -np.infty
    for  k in range(1,kmax):
        b = VBGMM(k,dim)
        b.guess_priors(x)
        b.initialize(x)
        b.estimate(x)
        ek = b.evidence(x)
        if verbose: print k,ek
        if ek > be:
            be = ek
            bestk = k
    assert(bestk<3)

def test_evidence(verbose=0,k=1):
    """
    Compare the evidence estimated by Chib's method
    with the variational evidence (free energy)
    fixme : this one really takes time
    """
    n=50
    dim=2
    x = nr.randn(n,dim)
    x[:15] += 3
    
    b = VBGMM(k,dim)
    b.guess_priors(x)
    b.initialize(x)
    b.estimate(x)
    vbe = b.evidence(x),
    if verbose:
        print 'vb: ',vbe, 

    niter = 1000
    b = BGMM(k,dim)
    b.guess_priors(x)
    b.initialize(x)
    b.sample(x,100)
    w,cent,prec,pz = b.sample(x, niter=niter, mem=1)
    bplugin =  BGMM(k, dim, cent, prec, w)
    bplugin.guess_priors(x)
    bfchib = bplugin.bayes_factor(x, pz.astype(np.int), 1)
    if verbose:
        print ' chib:', bfchib
    assert(bfchib>vbe)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


