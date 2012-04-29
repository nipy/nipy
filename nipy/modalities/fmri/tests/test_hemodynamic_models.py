import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, TestCase

from ..hemodynamic_models import (
    spm_hrf, spm_time_derivative, spm_dispersion_derivative,
    resample_regressor, _orthogonalize, sample_condition,
    _regressor_names, _hrf_kernel, glover_hrf, 
    glover_time_derivative, compute_regressor)


def test_spm_hrf():
    """ test that the spm_hrf is correctly normalized and has correct length
    """
    h = spm_hrf(2.0)
    assert_almost_equal(h.sum(), 1)
    assert len(h) == 256

def test_spm_hrf_derivative():
    """ test that the spm_hrf is correctly normalized and has correct length
    """
    h = spm_time_derivative(2.0)
    assert_almost_equal(h.sum(), 0)
    assert len(h) == 256
    h = spm_dispersion_derivative(2.0)
    assert_almost_equal(h.sum(), 0)
    assert len(h) == 256

def test_glover_hrf():
    """ test that the spm_hrf is correctly normalized and has correct length
    """
    h = glover_hrf(2.0)
    assert_almost_equal(h.sum(), 1)
    assert len(h) == 256

def test_glover_time_derivative():
    """ test that the spm_hrf is correctly normalized and has correct length
    """
    h = glover_time_derivative(2.0)
    assert_almost_equal(h.sum(), 0)
    assert len(h) == 256
    
def test_resample_regressor():
    """ test regressor resampling on a linear function
    """
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 30)
    z = resample_regressor(x, x, y)
    assert_almost_equal(z, y)

def test_resample_regressor_nl():
    """ test regressor resampling on a sine function
    """
    x = np.linspace(0, 10, 1000)
    y = np.linspace(0, 10, 30)
    z = resample_regressor(np.cos(x), x, y)
    assert_almost_equal(z, np.cos(y), decimal=2)

def test_orthogonalize():
    """ test that the orthogonalization is OK 
    """
    X = np.random.randn(100, 5)
    X = _orthogonalize(X)
    K = np.dot(X.T, X)
    K -= np.diag(np.diag(K))
    assert (K ** 2).sum() < 1.e-16

def test_orthogonalize_trivial():
    """ test that the orthogonalization is OK 
    """
    X = np.random.randn(100)
    Y = X.copy()
    X = _orthogonalize(X)
    assert (Y == X).all()

def test_sample_condition_1():
    """ Test that the experimental condition is correctly sampled
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = sample_condition(condition, frametimes, oversampling=1)
    assert reg.sum() == 3
    assert reg[1] == 1
    assert reg[37] == 1
    assert reg[20] ==1

def test_sample_condition_2():
    """ Test that the experimental condition is correctly sampled
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = sample_condition(condition, frametimes, oversampling=1)
    assert reg.sum() == 6
    assert reg[1] == 1
    assert reg[38] == 1
    assert reg[21] ==1

def test_sample_condition_3():
    """ Test that the experimental condition is correctly sampled
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = sample_condition(condition, frametimes, oversampling=10)
    assert_almost_equal(reg.sum(), 60.)
    assert reg[10] == 1
    assert reg[380] == 1
    assert reg[210] == 1
    assert np.sum(reg > 0) == 60
    
def test_sample_condition_4():
    """ Test that the experimental condition is correctly sampled
    with wrongly placed trials
    """
    condition = ([-3, 1, 20, 36.5, 51], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = sample_condition(condition, frametimes, oversampling=1)
    assert reg.sum() == 3
    assert reg[1] == 1
    assert reg[37] == 1
    assert reg[20] ==1

def test_sample_condition_5():
    """ Test that the experimental condition is correctly sampled
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1., -1., 5.])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = sample_condition(condition, frametimes, oversampling=1)
    assert reg.sum() == 10
    assert reg[1] == 1.
    assert reg[20] == -1.
    assert reg[37] == 5.

def test_names():
    """ Test the regressor naming function
    """
    name = 'con'
    assert _regressor_names(name, 'spm') == ['con']
    assert _regressor_names(name, 'spm_time') == ['con', 'con_derivative']
    assert _regressor_names(name, 'spm_time_dispersion') == \
        ['con', 'con_derivative', 'con_dispersion']
    assert _regressor_names(name, 'canonical') == ['con']
    assert _regressor_names(name, 'canonical with derivative') == \
        ['con', 'con_derivative']

def test_hkernel():
    """ test the hrf computation
    """
    tr = 2.0
    h = _hrf_kernel('spm', tr)
    assert_almost_equal(h[0], spm_hrf(tr))
    assert len(h) == 1
    h = _hrf_kernel('spm_time', tr)
    assert_almost_equal(h[1], spm_time_derivative(tr))
    assert len(h) == 2
    h = _hrf_kernel('spm_time_dispersion', tr)
    assert_almost_equal(h[2], spm_dispersion_derivative(tr))
    assert len(h) == 3
    h = _hrf_kernel('canonical', tr)
    assert_almost_equal(h[0], glover_hrf(tr))
    assert len(h) == 1
    h = _hrf_kernel('canonical with derivative', tr)
    assert_almost_equal(h[1], glover_time_derivative(tr))
    assert_almost_equal(h[0], glover_hrf(tr))
    assert len(h) == 2
    h = _hrf_kernel('fir', tr, fir_delays = np.arange(4))
    assert len(h) == 4
    for dh in h:
        assert dh.sum() == 16.
    
def test_make_regressor_1():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frametimes = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    reg, reg_names = compute_regressor(condition, hrf_model, frametimes)
    assert_almost_equal(reg.sum(), 6, 1)
    assert reg_names[0] == 'cond'

def test_make_regressor_2():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frametimes = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    reg, reg_names = compute_regressor(condition, hrf_model, frametimes)
    assert_almost_equal(reg.sum() * 16, 3, 1)
    assert reg_names[0] == 'cond'


def test_make_regressor_3():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frametimes = np.linspace(0, 138, 70)
    hrf_model = 'fir'
    reg, reg_names = compute_regressor(condition, hrf_model, frametimes, 
                                       fir_delays=np.arange(4))
    assert (np.unique(reg) == np.array([0, 1])).all()
    assert (np.sum(reg, 0) == np.array([3, 3, 3, 3])).all()
    assert len(reg_names) == 4

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
