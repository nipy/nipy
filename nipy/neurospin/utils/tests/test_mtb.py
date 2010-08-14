# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the roi utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

import numpy as np
from numpy.random import binomial
import nipy.neurospin.utils.two_binomial_mixture as tbm

def test_mtb1():
    from numpy.random import rand
    MB = tbm.TwoBinomialMixture()
    xmax = 10
    n = 1000
    x = np.concatenate(((xmax+1)*rand(n), rand(9*n))).astype(np.int)
    MB.EMalgo(x, xmax)
    MB.parameters()
    kappa = MB.kappa()
    assert (kappa>.7)*(kappa<.8)

def test_mtb2():
    xmax = 10
    n = 100
    x = binomial(xmax,0.3,n)
    MB = tbm.TwoBinomialMixture()
    MB.EMalgo(x,xmax)
    MB.parameters()
    kappa  = MB.kappa()
    print kappa
    assert(kappa<.2)
    
def test_mb3():
    xmax = 10
    n = 100
    x = np.concatenate((binomial(xmax,0.1,n),binomial(xmax,0.9,n)))
    MB = tbm.TwoBinomialMixture()
    MB.EMalgo(x,xmax)
    MB.parameters()
    assert (np.absolute(MB.Lambda-0.5)<0.1)
    
def test_mb4():
    xmax = 5
    n = 100
    x = np.concatenate((binomial(xmax, 0.1, n), binomial(xmax, 0.9, n)))
    MB = tbm.TwoBinomialMixture()
    MB.EMalgo(x, xmax)
    MB.parameters()
    assert np.absolute(MB.Lambda-0.5) < 0.1

def test_mb5():
    xmax = 3
    n = 1000
    x = np.concatenate((binomial(xmax,0.1,9*n),binomial(xmax,0.9,n)))
    MB = tbm.TwoBinomialMixture()
    MB.EMalgo(x,xmax)
    MB.parameters()
    assert(np.absolute(MB.Lambda-0.9)<0.1)

def test_mb6():
    xmax = 5
    n = 1000
    x = np.concatenate((binomial(xmax,0.05,9*n),binomial(xmax,0.5,n)))
    MB = tbm.TwoBinomialMixture()
    MB.EMalgo(x,xmax)
    MB.parameters()
    assert MB.r0<.1 

def test_mb7():
    xmax = 5
    n = 100
    x = np.concatenate((binomial(xmax,0.1,99*n),binomial(xmax,0.8,n)))
    MB = tbm.TwoBinomialMixture()
    MB.EMalgo(x,xmax)
    MB.parameters()
    assert MB.r1>0.7
    
if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

