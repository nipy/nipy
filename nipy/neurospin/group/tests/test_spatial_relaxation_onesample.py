import unittest

import numpy as np

import nipy.neurospin.group.spatial_relaxation_onesample as os

verbose = False

def make_data(n=10, dim=np.array([20,20,20]), r=5, amplitude=10, noise=1, jitter=None):
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
        X[i] = np.random.randn(p)
        o = np.array(dim/2)
        if jitter!=None:
            o += np.round(np.random.randn(3)*jitter).clip(r-dim/2,dim/2-r)
        Ii = XYZvol[o[0]-r:o[0]+r, o[1]-r:o[1]+r, o[2]-r:o[2]+r].ravel()
        X[i,Ii] += amplitude
        vardata[i] = np.square(np.random.randn(p))*noise**2
        data[i] = X[i] + np.random.randn(p)*np.sqrt(vardata[i])
    return data, XYZ, XYZvol, vardata, signal


class TestMultivariateStatSaem(unittest.TestCase):
    
    def test_evaluate_exact(self):
        # without mfx nor spatial relaxation
        data, XYZ, XYZvol, vardata, signal = make_data(n=20, dim=np.array([20,20,20]), r=3, amplitude=5, noise=0, jitter=0)
        p = len(signal)
        XYZvol *= 0
        XYZvol[list(XYZ)] = np.arange(p)
        P = os.multivariate_stat(data)
        P.init_hidden_variables()
        P.evaluate(nsimu=100, burnin=100, J=[XYZvol[5, 5, 5]], 
                   compute_post_mean=True, verbose=verbose)
        P.log_likelihood_values = P.compute_log_region_likelihood()
        # Verify code consistency
        Q = os.multivariate_stat(data, vardata*0, XYZ, std=0, sigma=5)
        Q.init_hidden_variables()
        Q.evaluate(nsimu=100, burnin=100, J = [XYZvol[5,5,5]], 
                   compute_post_mean=True, update_spatial=False, 
                   verbose=verbose)
        Q.log_likelihood_values = Q.compute_log_region_likelihood()
        self.assertAlmostEqual(P.mean_m.mean(), Q.mean_m.mean(), int(np.log10(P.nsimu))-1)
        self.assertAlmostEqual(Q.log_likelihood_values.sum(), P.log_likelihood_values.sum(), 1)
    
    def test_model_selection_exact(self):
        data, XYZ, XYZvol, vardata, signal = make_data(n=30, dim=10, r=3, amplitude=1, noise=0, jitter=0)
        labels = (signal > 0).astype(int)
        P1 = os.multivariate_stat(data, labels=labels)
        P1.init_hidden_variables()
        P1.evaluate(nsimu=100, burnin=10, verbose=verbose)
        L1 = P1.compute_log_region_likelihood()
        Prior1 = P1.compute_log_prior()
        Post1 = P1.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)
        M1 = L1 + Prior1[:-1] - Post1
        self.assertAlmostEqual(M1.mean(), 
                               P1.compute_marginal_likelihood().mean(), 0)
        P0 = os.multivariate_stat(data, labels=labels)
        P0.network *= 0
        P0.init_hidden_variables()
        P0.evaluate(nsimu=100, burnin=100, verbose=verbose)
        L0 = P0.compute_log_region_likelihood()
        Prior0 = P0.compute_log_prior()
        Post0 = P0.compute_log_posterior(nsimu=1e2, burnin=1e2, 
                                         verbose=verbose)
        M0 = L0 + Prior0[:-1] - Post0
        self.assertAlmostEqual(M0.mean(), 
                               P0.compute_marginal_likelihood().mean(), 0)
        self.assertTrue(M1[1] > M0[1])
        self.assertTrue(M1[0] < M0[0])
    
    def test_model_selection_mfx_spatial_rand_walk(self):
        data, XYZ, XYZvol, vardata, signal = make_data(n=20, dim=np.array([1,20,20]), 
                                                r=3, amplitude=3, noise=1, jitter=0.5)
        labels = (signal > 0).astype(int)
        P = os.multivariate_stat(data, vardata, XYZ, std=0.5, sigma=5, labels=labels)
        P.network[:] = 0
        P.init_hidden_variables()
        P.evaluate(nsimu=100, burnin=100, verbose=verbose, 
                    proposal='rand_walk', proposal_std=0.5)
        L00 = P.compute_log_region_likelihood()
        # Test simulated annealing procedure
        P.estimate_displacements_SA(nsimu=100, c=0.99, 
            proposal_std=P.proposal_std, verbose=verbose)
        L0 = P.compute_log_region_likelihood()
        self.assertTrue(L0.sum() > L00.sum())
        Prior0 = P.compute_log_prior()
        Post0 = P.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)
        M0 = L0 + Prior0[:-1] - Post0
        #self.assertAlmostEqual(0.1*M0.sum(), 0.1*P.compute_marginal_likelihood(verbose=verbose).sum(), 0)
        P.network[:] = 1
        P.init_hidden_variables(init_spatial=False)
        P.evaluate(nsimu=100, burnin=100, verbose=verbose, 
                    update_spatial=False)
        L1 = P.compute_log_region_likelihood()
        Prior1 = P.compute_log_prior()
        Post1 = P.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)
        M1 = L1 + Prior1[:-1] - Post1
        #self.assertAlmostEqual(0.1*M1.sum(), 0.1*P.compute_marginal_likelihood(verbose=verbose).sum(), 0)
        self.assertTrue(M1[1] > M0[1])
        self.assertTrue(M1[0] < M0[0])



class TestMultivariateStatMcmc(unittest.TestCase):
    
    def test_evaluate_mfx_spatial(self):
        data, XYZ, XYZvol, vardata, signal = make_data(n=20, 
                    dim=10, r=3, amplitude=5, noise=1, jitter=1)
        P = os.multivariate_stat(data, vardata, XYZ, std=1, sigma=3)
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
        self.assertAlmostEqual(-L1.sum(), L2.sum()*2, 2)
        # Test posterior density computation
        #Prior = P.compute_log_prior(v, m_mean, m_var)
        #Post = P.compute_log_posterior(v, m_mean, m_var, nsimu=10, 
                                       #burnin=10, verbose=verbose)

class TestLabelsPrior(unittest.TestCase):
     
    def test_update_labels(self):
        data, XYZ, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, amplitude=5, 
                        noise=1, jitter=1)
        P = os.multivariate_stat(data, vardata, XYZ)
        P.init_hidden_variables()
        p = P.data.shape[1]
        P.labels_prior = np.ones((1, p), float)
        P.label_values = np.zeros((1, p), int)
        P.labels_prior_mask = np.arange(p)
        P.update_labels()
        self.assertEqual(max(abs(P.labels - np.zeros(p, int))), 0)

if __name__ == "__main__":
    unittest.main()
