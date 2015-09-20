#!/usr/bin/env python
from __future__ import absolute_import

import numpy as np

from ..glm import glm

from numpy.testing import assert_almost_equal, TestCase

class TestFitting(TestCase):

    def make_data(self):
        dimt = 100
        dimx = 10
        dimy = 11
        dimz = 12 
        self.y = np.random.randn(dimt, dimx, dimy, dimz)
        X = np.array([np.ones(dimt), list(range(dimt))])
        self.X = X.transpose() ## the design matrix X must have dimt lines

    def ols(self, axis):
        y = np.rollaxis(self.y, 0, axis+1) ## time index is axis
        X = self.X
        m = glm(y, X, axis=axis)
        m1 = glm(y, X, axis=axis, method='kalman')
        b = m.beta
        b1 = m1.beta
        tcon = m.contrast([1,0])
        tcon1 = m1.contrast([1,0])
        z = tcon.zscore()
        z1 = tcon1.zscore()
        assert_almost_equal(b, b1)            
        ##assert_almost_equal(v, v1, decimal=2)
        ##assert_almost_equal(z, z1, decimal=3)
        
    def test_ols_axis0(self):
        self.make_data()
        self.ols(0)
        
    def test_ols_axis1(self):
        self.make_data()
        self.ols(1)

    def test_ols_axis2(self):
        self.make_data()
        self.ols(2)

    def test_ols_axis3(self):
        self.make_data()
        self.ols(3)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
