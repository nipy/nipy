"""
Test the Bayesian GMM.

fixme : some of these tests take too much time at the moment
to be real unit tests

Author : Bertrand Thirion, 2009 
"""

#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

import numpy as np
import numpy.random as nr
from nipy.neurospin.clustering.bgmm import *


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
    b.init(x)
    b.sample(x,1)
    w,cent,prec,pz = b.sample(x,niter,mem=1)
    b.plugin(cent,prec,w)
    z = pz[:,0]
    
    # fixme : find a less trivial test
    assert(z.max()+1==b.k)

def test_gmm_bf(kmax=5,verbose = 0):
    """
    perform a model selection procedure on a  gmm
    with Bayes factor estimations
    kmax : range of values that are tested
    """
    n=100
    dim=2
    x = nr.randn(n,dim)
    #x[:30] += 2
    niter = 1000

    bbf = -np.infty
    for k in range(1,kmax):
        b = BGMM(k,dim)
        b.guess_priors(x)
        b.init(x)
        b.sample(x,100)
        w,cent,prec,pz = b.sample(x,niter=niter,mem=1)
        bplugin =  BGMM(k,dim,cent,prec,w)
        bplugin.guess_priors(x)
        bfk = bplugin.Bfactor(x,pz.astype(np.int),1)
        if verbose:
            print k, bfk
        if bfk>bbf:
            bestk = k
            bbf = bfk
    assert(bestk<4)
    

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
    b.init(x)
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
    show = 0
    be = -np.infty
    for  k in range(1,kmax):
        b = VBGMM(k,dim)
        b.guess_priors(x)
        b.init(x)
        b.estimate(x)
        z = b.map_label(x)
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
    n=100
    dim=2
    x = nr.randn(n,dim)
    x[:30] += 3
    show = 0
    
    b = VBGMM(k,dim)
    b.guess_priors(x)
    b.init(x)
    b.estimate(x)
    vbe = b.evidence(x),
    if verbose:
        print 'vb: ',vbe, 

    niter = 1000
    b = BGMM(k,dim)
    b.guess_priors(x)
    b.init(x)
    b.sample(x,100)
    w,cent,prec,pz = b.sample(x,niter=niter,mem=1)
    bplugin =  BGMM(k,dim,cent,prec,w)
    bplugin.guess_priors(x)
    bfchib = bplugin.Bfactor(x,pz.astype(np.int),1)
    if verbose:
        print ' chib:', bfchib
    assert(bfchib>vbe)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


