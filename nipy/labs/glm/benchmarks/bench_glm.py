from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from ..glm import glm

def make_data():
    dimt = 100
    dimx = 10
    dimy = 11
    dimz = 12 
    y = np.random.randn(dimt, dimx, dimy, dimz)
    X = np.array([np.ones(dimt), list(range(dimt))])
    X = X.transpose() ## the design matrix X must have dimt lines
    return y, X

def ols(axis, y, X):
    y = np.rollaxis(y, 0, axis+1) ## time index is axis
    X = X
    m = glm(y, X, axis=axis)
    m1 = glm(y, X, axis=axis, method='kalman')
    b = m.beta
    b1 = m1.beta
    v = m.s2
    v1 = m1.s2
    print("Comparing standard OLS with Kalman OLS...")
    re = ( np.abs(b-b1) / (np.abs(b)+1e-20) ).mean()
    print("  Relative difference in Effect estimate: %s" % re)
    re = ( np.abs(v-v1) / (np.abs(v)+1e-20) ).mean()
    print("  Relative difference in Variance: %s" % re)
    tcon = m.contrast([1,0])
    tcon1 = m1.contrast([1,0])
    z = tcon.zscore()
    z1 = tcon1.zscore()
    re = ( abs(z-z1) / (abs(z)+1e-20) ).mean()
    print("  Relative difference in z score: %s" % re)

def bench_ols_axis0():
    x, Y = make_data()
    ols(0, x, Y)

def bench_ols_axis1():
    x, Y = make_data()
    ols(1, x, Y)

def bench_ols_axis2():
    x, Y = make_data()
    ols(2, x, Y)

def bench_ols_axis3():
    x, Y = make_data()
    ols(3, x, Y)

