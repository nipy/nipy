#!/usr/bin/env python

import numpy as np
from scipy import special

from ..routines import median, mahalanobis, gamln, psi

from nose.tools import assert_true
from numpy.testing import assert_almost_equal, assert_equal, TestCase


class TestAll(TestCase):

    def test_median(self):
        x = np.random.rand(100)
        assert_almost_equal(median(x), np.median(x))

    def test_median2(self):
        x = np.random.rand(101)
        assert_equal(median(x), np.median(x))

    def test_median3(self):
        x = np.random.rand(10, 30, 11)
        assert_almost_equal(np.squeeze(median(x,axis=1)), np.median(x,axis=1))

    def test_mahalanobis(self):
        x = np.random.rand(100) / 100
        A = np.random.rand(100, 100) / 100
        A = np.dot(A.transpose(), A) + np.eye(100)
        mah = np.dot(x, np.dot(np.linalg.inv(A), x))
        assert_almost_equal(mah, mahalanobis(x, A), decimal=1) 

    def test_mahalanobis2(self):
        x = np.random.rand(100,3,4)
        Aa = np.zeros([100,100,3,4])
        for i in range(3):
            for j in range(4):
                A = np.random.rand(100,100)
                A = np.dot(A.T, A)
                Aa[:,:,i,j] = A
        i = np.random.randint(3)
        j = np.random.randint(4)
        mah = np.dot(x[:,i,j], np.dot(np.linalg.inv(Aa[:,:,i,j]), x[:,i,j]))
        f_mah = (mahalanobis(x, Aa))[i,j]
        assert_true(np.allclose(mah, f_mah))

    def test_gamln(self):
        for x in (0.01+100*np.random.random(50)):
            scipy_gamln = special.gammaln(x)
            my_gamln = gamln(x)
            assert_almost_equal(scipy_gamln, my_gamln)

    def test_psi(self):
        for x in (0.01+100*np.random.random(50)):
            scipy_psi = special.psi(x)
            my_psi = psi(x)
            assert_almost_equal(scipy_psi, my_psi)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
