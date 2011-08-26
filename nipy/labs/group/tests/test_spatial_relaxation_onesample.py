# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from ..spatial_relaxation_onesample import multivariate_stat

from ....testing import (dec, assert_true, assert_equal, assert_almost_equal)

verbose = False

def make_data(n=10, dim=np.array([20,20,20]), r=5, amplitude=10, noise=1,
              jitter=None, prng=np.random):
    if np.isscalar(dim):
        dim = np.array([dim,dim,dim])
    XYZvol = np.zeros((dim),int)
    XYZ = np.array(np.where(XYZvol==0))
    p = XYZ.shape[1]
    #mask = np.arange(p)
    XYZvol[XYZ[0],XYZ[1],XYZ[2]] = np.arange(p)
    o = np.array(dim/2)
    maskvol = np.zeros(dim, int)
    maskvol[list(XYZ)] = np.arange(p)
    Isignal = maskvol[dim[0]/2-r:dim[0]/2+r, dim[1]/2-r:dim[1]/2+r, dim[2]/2-r:dim[2]/2+r].ravel()
    signal = np.zeros(p, float)
    signal[Isignal] += amplitude
    X = np.zeros((n, p), float) + np.nan
    data = np.zeros((n, p), float) + np.nan
    vardata = np.zeros((n, p), float) + np.nan
    for i in xrange(n):
        X[i] = prng.randn(p)
        o = np.array(dim/2)
        if jitter!=None:
            # Not in place to avoid stricter numpy 2 casting
            o = o + np.round(prng.randn(3)*jitter).clip(r-dim/2,dim/2-r)
        Ii = XYZvol[o[0]-r:o[0]+r, o[1]-r:o[1]+r, o[2]-r:o[2]+r].ravel()
        X[i,Ii] += amplitude
        vardata[i] = np.square(prng.randn(p))*noise**2
        data[i] = X[i] + prng.randn(p)*np.sqrt(vardata[i])
    return data, XYZ, XYZvol, vardata, signal


def test_evaluate_exact():
    # without mfx nor spatial relaxation
    prng = np.random.RandomState(10)
    data, XYZ, XYZvol, vardata, signal = make_data(n=20,
            dim=np.array([20,20,20]), r=3, amplitude=5, noise=0,
            jitter=0, prng=prng)
    p = len(signal)
    XYZvol *= 0
    XYZvol[list(XYZ)] = np.arange(p)
    P = multivariate_stat(data)
    P.init_hidden_variables()
    P.evaluate(nsimu=100, burnin=100, J=[XYZvol[5, 5, 5]],
               compute_post_mean=True, verbose=verbose)
    P.log_likelihood_values = P.compute_log_region_likelihood()
    # Verify code consistency
    Q = multivariate_stat(data, vardata*0, XYZ, std=0, sigma=5)
    Q.init_hidden_variables()
    Q.evaluate(nsimu=100, burnin=100, J = [XYZvol[5,5,5]],
               compute_post_mean=True, update_spatial=False,
               verbose=verbose)
    Q.log_likelihood_values = Q.compute_log_region_likelihood()
    assert_almost_equal(P.mean_m.mean(),
                              Q.mean_m.mean(),
                              int(np.log10(P.nsimu))-1)
    assert_almost_equal(Q.log_likelihood_values.sum(),
                              P.log_likelihood_values.sum(), 0)


def test_model_selection_exact():
    prng = np.random.RandomState(10)
    data, XYZ, XYZvol, vardata, signal = make_data(n=30, dim=20, r=3,
                amplitude=1, noise=0, jitter=0, prng=prng)
    labels = (signal > 0).astype(int)
    P1 = multivariate_stat(data, labels=labels)
    P1.init_hidden_variables()
    P1.evaluate(nsimu=100, burnin=10, verbose=verbose)
    L1 = P1.compute_log_region_likelihood()
    Prior1 = P1.compute_log_prior()
    #v, m_mean, m_var = P1.v.copy(), P1.m_mean.copy(), P1.m_var.copy()
    Post1 = P1.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)
    M1 = L1 + Prior1[:-1] - Post1[:-1]
    assert_almost_equal(M1.mean(), P1.compute_marginal_likelihood().mean(), 0)
    P0 = multivariate_stat(data, labels=labels)
    P0.network *= 0
    P0.init_hidden_variables()
    P0.evaluate(nsimu=100, burnin=100, verbose=verbose)
    L0 = P0.compute_log_region_likelihood()
    Prior0 = P0.compute_log_prior()
    Post0 = P0.compute_log_posterior(nsimu=1e2, burnin=1e2,
                                     verbose=verbose)
    M0 = L0 + Prior0[:-1] - Post0[:-1]
    assert_almost_equal(M0.mean(), P0.compute_marginal_likelihood().mean(), 0)
    assert_true(M1[1] > M0[1])
    assert_true(M1[0] < M0[0])


@dec.slow # test takes around 7 minutes on my (MB) Mac laptop
def test_model_selection_mfx_spatial_rand_walk():
    prng = np.random.RandomState(10)
    data, XYZ, XYZvol, vardata, signal = make_data(
        n=20,
        dim=np.array([1,20,20]),
        r=3, amplitude=3, noise=1,
        jitter=0.5, prng=prng)
    labels = (signal > 0).astype(int)
    P = multivariate_stat(data, vardata, XYZ, std=0.5, sigma=5, labels=labels)
    P.network[:] = 0
    P.init_hidden_variables()
    P.evaluate(nsimu=100, burnin=100, verbose=verbose,
               proposal='rand_walk', proposal_std=0.5)
    L00 = P.compute_log_region_likelihood()
    # Test simulated annealing procedure
    P.estimate_displacements_SA(nsimu=100, c=0.99,
                                proposal_std=P.proposal_std, verbose=verbose)
    L0 = P.compute_log_region_likelihood()
    assert_true(L0.sum() > L00.sum())
    #Prior0 = P.compute_log_prior()
    #Post0 = P.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)
    #M0 = L0 + Prior0[:-1] - Post0[:-1]
    M0 = P.compute_marginal_likelihood(update_spatial=True)
    #yield assert_almost_equal(M0.sum(), P.compute_marginal_likelihood(verbose=verbose).sum(), 0)
    P.network[1] = 1
    #P.init_hidden_variables(init_spatial=False)
    P.init_hidden_variables(init_spatial=False)
    P.evaluate(nsimu=100, burnin=100, verbose=verbose,
               update_spatial=False, proposal_std=P.proposal_std)
    #L1 = P.compute_log_region_likelihood()
    #Prior1 = P.compute_log_prior()
    #Post1 = P.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)
    #M1 = L1 + Prior1[:-1] - Post1[:-1]
    M1 = P.compute_marginal_likelihood(update_spatial=True)
    #yield assert_almost_equal(0.1*M1.sum(), 0.1*P.compute_marginal_likelihood(verbose=verbose).sum(), 0)
    assert_true(M1 > M0)


def test_evaluate_mfx_spatial():
    prng = np.random.RandomState(10)
    data, XYZ, XYZvol, vardata, signal = make_data(
        n=20,
        dim=10, r=3, amplitude=5, noise=1, jitter=1,
        prng=prng)
    P = multivariate_stat(data, vardata, XYZ, std=1, sigma=3)
    P.init_hidden_variables()
    P.evaluate(nsimu=5, burnin=5,
               J=[P.D.XYZ_vol[10, 10, 10]],
               verbose=verbose, mode='mcmc')
    # Test log_likelihood computation
    v = P.v.copy()
    m_var = P.m_var.copy()
    m_mean = P.m_mean.copy()
    L1 = P.compute_log_region_likelihood_slow(v, m_mean, m_var)
    L2 = P.compute_log_region_likelihood(v, m_mean, m_var)
    assert_almost_equal(-L1.sum(), L2.sum()*2, 2)
    # Test posterior density computation
    #Prior = P.compute_log_prior(v, m_mean, m_var)
    #Post = P.compute_log_posterior(v, m_mean, m_var, nsimu=10, 
                      #burnin=10, verbose=verbose)


def test_update_labels():
    prng = np.random.RandomState(10)
    data, XYZ, XYZvol, vardata, signal = make_data(
        n=20, dim=20, r=3,
        amplitude=5, noise=1, jitter=1, prng=prng)
    P = multivariate_stat(data, vardata, XYZ)
    P.init_hidden_variables()
    p = P.data.shape[1]
    P.labels_prior = np.ones((1, p), float)
    P.label_values = np.zeros((1, p), int)
    P.labels_prior_mask = np.arange(p)
    P.update_labels()
    assert_equal(max(abs(P.labels - np.zeros(p, int))), 0)
