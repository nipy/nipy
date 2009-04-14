#!/usr/bin/env python

from numpy.testing import assert_almost_equal, assert_equal, TestCase

from numpy import median, dot, squeeze, zeros
import numpy as np
from numpy.random import rand, randint
from numpy.linalg import inv
from scipy import special

import nipy.neurospin.utils as fu
import nipy.neurospin.utils.routines as routines

class TestAll(TestCase):

    def test_median(self):
        x = rand(100)
        assert_almost_equal(fu.median(x), median(x))

    def test_median2(self):
        x = rand(101)
        assert_equal(fu.median(x), median(x))

    def test_median3(self):
        x = rand(10,30,11)
        assert_almost_equal(squeeze(fu.median(x,axis=1)), median(x,axis=1))

    def test_mahalanobis(self):
        x = rand(100)
        A = rand(100,100)
        A = dot(A.transpose(), A)
        mah = dot(x, dot(inv(A), x))
        assert_almost_equal(mah, fu.mahalanobis(x, A), decimal=4) 
        
    def test_mahalanobis2(self):
        x = rand(100,3,4)
        Aa = zeros([100,100,3,4])
        for i in range(3):
            for j in range(4):
                A = rand(100,100)
                A = dot(A.transpose(), A)
                Aa[:,:,i,j] = A
        i = randint(3)
        j = randint(4)
        mah = dot(x[:,i,j], dot(inv(Aa[:,:,i,j]), x[:,i,j]))
        f_mah = (fu.mahalanobis(x, Aa))[i,j]        
        assert_almost_equal(mah, f_mah, decimal=3)
       
    def test_gamln(self):
        for x in (0.01+100*np.random.random(50)):
            scipy_gamln = special.gammaln(x)
            my_gamln = routines.gamln(x)
            assert_almost_equal(scipy_gamln, my_gamln)

    def test_psi(self):
        for x in (0.01+100*np.random.random(50)):
            scipy_psi = special.psi(x)
            my_psi = routines.psi(x)
            assert_almost_equal(scipy_psi, my_psi)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
