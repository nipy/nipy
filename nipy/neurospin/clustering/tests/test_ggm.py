#!/usr/bin/env python

import numpy as np
import numpy.random as nr
from nipy.neurospin.clustering.ggmixture import *
import scipy.stats as st

def test_GGM1():
    shape = 1 
    scale = 1
    mean = 0
    var = 1
    G = GGM(shape,scale,mean,var)
    sx = 1000
    x = -2.5 + nr.randn(sx)
    G.estimate(x)
    G.parameters()
    b = np.absolute(G.mean+2.5)<0.5
    assert(b)

def test_GGM2():
    shape = 1 
    scale = 1
    mean = 0
    var = 1
    G = GGM(shape,scale,mean,var)
    sx = 1000
    x = -2.5 + nr.randn(sx)
    G.estimate(x)
    G.parameters()
    b = np.absolute(G.mixt)<0.1
    assert(b)

def test_GGGM0():
    G = GGGM()
    sx = 10000
    x = np.array([float(st.t.rvs(5)) for i in range(sx)])
    G.init(x)
    G.estimate(x)
    G.parameters()
    assert(np.absolute(G.mean)<0.1)

def test_GGGM1():
    G = GGGM()
    sx = 10000
    x = np.array([float(st.t.rvs(5)) for i in range(sx)])
    G.init_fdr(x)
    G.estimate(x)
    G.parameters()
    assert(np.absolute(G.mean)<0.1)
    
def test_GGGM2():
    G = GGGM()
    sx = 10000
    x = nr.randn(sx)
    G.init_fdr(x)
    G.estimate(x)
    assert(G.mixt[1]>0.9)

def test_Gamma_parameters1():
    import numpy.random as nr
    n = 1000
    X = nr.gamma(11., 3., n)
    G = Gamma()
    G.estimate(X)
    G.parameters()
    assert(np.absolute(G.shape-11)<1)

def test_Gamma_parameters2():
    import numpy.random as nr
    n = 1000
    X = nr.gamma(11., 3., n)
    G = Gamma()
    G.estimate(X)
    G.parameters()
    assert(np.absolute(G.scale-3)<0.5)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


