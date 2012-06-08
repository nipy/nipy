#!/usr/bin/env python

import numpy as np
import numpy.random as nr
from ..ggmixture import GGGM, GGM, Gamma
import scipy.stats as st

def test_GGM1(verbose=0):
    shape = 1 
    scale = 1
    mean = 0
    var = 1
    G = GGM(shape,scale,mean,var)
    sx = 1000
    x = -2.5 + nr.randn(sx)
    G.estimate(x)
    b = np.absolute(G.mean+2.5)<0.5
    if verbose:
        #G.parameters()
        print x.max()
    assert(b)

def test_GGM2(verbose=0):
    shape = 1 
    scale = 1
    mean = 0
    var = 1
    G = GGM(shape,scale,mean,var)
    sx = 1000
    x = -2.5 + nr.randn(sx)
    G.estimate(x)
    if verbose:
        G.parameters()
    b = np.absolute(G.mixt)<0.1
    assert(b)

def test_GGGM0(verbose=0, seed=1):
    G = GGGM()
    sx = 1000
    #x = np.array([float(st.t.rvs(dof)) for i in range(sx)])
    if seed:
        nr = np.random.RandomState([seed])
    else:
        import numpy.random as nr
    x = nr.randn(sx)
    G.init(x)
    G.estimate(x)
    if verbose:
        G.parameters()
    assert(np.absolute(G.mean)<0.3)

def test_GGGM1(verbose=0):
    G = GGGM()
    sx = 10000
    x = np.array([float(st.t.rvs(5)) for i in range(sx)])
    G.init_fdr(x)
    G.estimate(x)
    if verbose:
        G.parameters()
    assert(np.absolute(G.mean)<0.1)
    
def test_GGGM2(verbose=0):
    G = GGGM()
    sx = 10000
    x = nr.randn(sx)
    G.init_fdr(x)
    G.estimate(x)
    assert(G.mixt[1]>0.9)

def test_GGGM3(verbose=0):
    G = GGGM()
    sx = 1000
    x = 100 + np.array([float(st.t.rvs(5)) for i in range(sx)])
    G.init(x)
    G.estimate(x)
    if verbose:
        G.parameters()
    assert(np.absolute(G.mixt[0])<1.e-15)

def test_gamma_parameters1(verbose=0):
    import numpy.random as nr
    n = 1000
    X = nr.gamma(11., 3., n)
    G = Gamma()
    G.estimate(X)
    if verbose:
        G.parameters()
    assert(np.absolute(G.shape-11)<2.)

def test_gamma_parameters2(verbose=0):
    import numpy.random as nr
    n = 1000
    X = nr.gamma(11., 3., n)
    G = Gamma()
    G.estimate(X)
    if verbose:
        G.parameters()
    assert(np.absolute(G.scale-3)<0.5)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


