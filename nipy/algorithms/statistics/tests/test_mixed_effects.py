""" Testing the glm module
"""

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import assert_true
import numpy.random as nr

from ..mixed_effects_stat import (
    one_sample_ttest, one_sample_ftest, two_sample_ttest, two_sample_ftest, 
    generate_data, t_stat, mfx_stat)


def test_mfx():
    """ Test the generic mixed-effects model"""
    n_samples, n_tests = 20, 100
    np.random.seed(1)
    
    # generate some data
    V1 = np.random.rand(n_samples, n_tests)
    Y = generate_data(np.ones((n_samples, 1)), 0, 1, V1)
    X = np.random.randn(20, 3)

    # compute the test statistics
    t1, = mfx_stat(Y, V1, X, 1,return_t=True, 
                   return_f=False, return_effect=False, 
                   return_var=False)
    assert_true(t1.shape == (n_tests,))
    assert_true(t1.mean() < 5 / np.sqrt(n_tests))
    assert_true((t1.var() < 2) and (t1.var() > .5))
    t2, = mfx_stat(Y, V1, X * np.random.rand(3), 1)
    assert_almost_equal(t1, t2)
    f, = mfx_stat(Y, V1, X, 1, return_t=False, return_f=True)
    assert_almost_equal(t1 ** 2, f)
    v2, = mfx_stat(Y, V1, X, 1, return_t=False, return_var=True)
    assert_true((v2 > 0).all())
    fx, = mfx_stat(Y, V1, X, 1, return_t=False, return_effect=True)
    assert_true(fx.shape == (n_tests,))

def test_t_test():
    """ test that the t test run
    """
    n_samples, n_tests = 15, 100
    data = nr.randn(n_samples, n_tests)
    t = t_stat(data)
    assert_true(t.shape == (n_tests,))
    assert_true( np.abs(t.mean() < 5 / np.sqrt(n_tests)))
    assert_true(t.var() < 2)
    assert_true( t.var() > .5)
    
def test_two_sample_ttest():
    """ test that the mfx ttest indeed runs
    """
    n_samples, n_tests = 15, 4
    np.random.seed(1)
    
    # generate some data
    vardata = np.random.rand(n_samples, n_tests)
    data = generate_data(np.ones(n_samples), 0, 1, vardata)
        
    # compute the test statistics
    u = np.concatenate((np.ones(5), np.zeros(10)))
    t2 = two_sample_ttest(data, vardata, u, n_iter=5)
    assert t2.shape == (n_tests,)
    assert np.abs(t2.mean() < 5 / np.sqrt(n_tests))
    assert t2.var() < 2
    assert t2.var() > .5
    
    # try verbose mode
    t3 = two_sample_ttest(data, vardata, u, n_iter=5, verbose=1)
    assert_almost_equal(t2, t3)

def test_two_sample_ftest():
    """ test that the mfx ttest indeed runs
    """
    n_samples, n_tests = 15, 4
    np.random.seed(1)
    
    # generate some data
    vardata = np.random.rand(n_samples, n_tests)
    data = generate_data(np.ones((n_samples, 1)), 0, 1, vardata)
        
    # compute the test statistics
    u = np.concatenate((np.ones(5), np.zeros(10)))
    t2 = two_sample_ftest(data, vardata, u, n_iter=5)
    assert t2.shape == (n_tests,)
    assert np.abs(t2.mean() < 5 / np.sqrt(n_tests))
    assert t2.var() < 2
    assert t2.var() > .5
    
    # try verbose mode
    t3 = two_sample_ftest(data, vardata, u, n_iter=5, verbose=1)
    assert_almost_equal(t2, t3)

def test_mfx_ttest():
    """ test that the mfx ttest indeed runs
    """
    n_samples, n_tests = 15, 100
    np.random.seed(1)
    
    # generate some data
    vardata = np.random.rand(n_samples, n_tests)
    data = generate_data(np.ones((n_samples, 1)), 0, 1, vardata)
        
    # compute the test statistics
    t2 = one_sample_ttest(data, vardata, n_iter=5)
    assert t2.shape == (n_tests,)
    assert np.abs(t2.mean() < 5 / np.sqrt(n_tests))
    assert t2.var() < 2
    assert t2.var() > .5
    
    # try verbose mode
    t3 = one_sample_ttest(data, vardata, n_iter=5, verbose=1)
    assert_almost_equal(t2, t3)

def test_mfx_ftest():
    """ test that the mfx ftest indeed runs
    """
    n_samples, n_tests = 15, 100
    np.random.seed(1)
    
    # generate some data
    vardata = np.random.rand(n_samples, n_tests)
    data = generate_data(np.ones((n_samples, 1)), 0, 1, vardata)
        
    # compute the test statistics
    f = one_sample_ftest(data, vardata, n_iter=5)
    assert f.shape == (n_tests,)
    assert (np.abs(f.mean() - 1) < 1)
    assert f.var() < 10
    assert f.var() > .2
     

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

