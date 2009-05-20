import unittest

import numpy as np
import scipy.stats as st

import nipy.neurospin.group.spatial_relaxation_onesample as os

verbose = False

def make_data(n=10, dim=20, r=5, mdim=15, maskdim=20, amplitude=10, noise=1, jitter=None, activation=False):
    XYZvol = np.zeros((dim,dim,dim),int)
    XYZ = np.array(np.where(XYZvol==0))
    p = XYZ.shape[1]
    #mask = np.arange(p)
    XYZvol[XYZ[0],XYZ[1],XYZ[2]] = np.arange(p)
    o = np.array([dim/2,dim/2,dim/2])
    I = XYZvol[(dim-mdim)/2:(dim+mdim)/2,(dim-mdim)/2:(dim+mdim)/2,(dim-mdim)/2:(dim+mdim)/2].ravel()
    mask = XYZvol[ (dim-maskdim)/2 : (dim+maskdim)/2, (dim-maskdim)/2 : (dim+maskdim)/2, (dim-maskdim)/2 : (dim+maskdim)/2 ].ravel()
    q = len(mask)
    maskvol = np.zeros((dim,dim,dim),int)
    maskvol[XYZ[0,mask],XYZ[1,mask],XYZ[2,mask]] = np.arange(q)
    Isignal = maskvol[dim/2-r:dim/2+r,dim/2-r:dim/2+r,dim/2-r:dim/2+r].ravel()
    signal = np.zeros(q,float)
    signal[Isignal] += amplitude
    X = np.zeros((n,p),float) + np.nan
    data = np.zeros((n,p),float) + np.nan
    vardata = np.zeros((n,p),float) + np.nan
    for i in xrange(n):
        X[i,I] = np.random.randn(len(I))
        if activation:
            o = np.array([dim/2,dim/2,dim/2])
            if jitter!=None:
                o += np.round(np.random.randn(3)*jitter).clip(r-mdim/2,mdim/2-r)
            #print o
        Ii = XYZvol[o[0]-r:o[0]+r,o[1]-r:o[1]+r,o[2]-r:o[2]+r].ravel()
        X[i,Ii] += amplitude
        vardata[i,I] = np.square(np.random.randn(len(I)))*noise**2
        data[i,I] = X[i,I] + np.random.randn(len(I))*np.sqrt(vardata[i,I])
    return data, XYZ, mask, XYZvol, vardata, signal


class test_multivariate_stat_saem(unittest.TestCase):
    
    def test_evaluate_exact(self):
        # without mfx nor spatial relaxation
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=0, jitter=0, activation=False)
        #verbose=False
        XYZvol *= 0
        XYZvol[XYZ[0,mask],XYZ[1,mask],XYZ[2,mask]] = np.arange(len(mask))
        ##P = os.stat(data, XYZ, mask, mean_prior_rate = 0, var_prior_scale=0, var_prior_shape=-0.5)
        P = os.multivariate_stat(data[:, mask])
        P.init_hidden_variables()
        P.evaluate(nsimu=100, burnin=100, J = [XYZvol[5,5,5]], compute_post_mean=True, verbose=verbose)
        P.log_likelihood_values = P.compute_log_region_likelihood()
        # Verify code consistency
        Q = os.multivariate_stat(data[:, mask], vardata[:, mask] * 0, XYZ[:, mask], std=0, sigma=5)
        Q.init_hidden_variables()
        Q.evaluate(nsimu=100, burnin=100, J = [XYZvol[5,5,5]], compute_post_mean=True, update_spatial=False, verbose=verbose)
        Q.log_likelihood_values = Q.compute_log_region_likelihood()
        self.assertAlmostEqual(P.mean_m.mean(), Q.mean_m.mean(), int(np.log10(P.nsimu))-1)
        self.assertAlmostEqual(Q.log_likelihood_values.sum(), P.log_likelihood_values.sum(), 1)
    
    def test_evaluate_mfx_spatial_fixed(self):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        #verbose=True
        P = os.multivariate_stat(data[:, mask], vardata[:, mask], XYZ[:, mask], std=1, sigma=3)
        P.init_hidden_variables()
        P.evaluate(nsimu=10, burnin=10, J = [P.D.XYZ_vol[10,10,10]],verbose=verbose, compute_post_mean=True)
        proposal_mean = P.mean_U
        proposal_std = np.sqrt(np.clip(P.var_U, 1e-4, 4))
        P.evaluate(10, 0, [P.D.XYZ_vol[10,10,10]], False, 'fixed', proposal_std, proposal_mean)
        self.assertEqual(max(abs(P.labels - np.zeros(len(mask), int))), 0)
    
    def test_model_selection_exact(self):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=30, dim=30, r=3, mdim=10, maskdim=10, amplitude=1, noise=1, jitter=1, activation=False)
        #verbose=False
        P1 = os.multivariate_stat(data[:, mask])
        P1.init_hidden_variables()
        P1.evaluate(nsimu=100, burnin=10, verbose=verbose)
        L1 = P1.compute_log_region_likelihood()[0]
        Prior1 = P1.compute_log_prior()[0]
        Post1 = P1.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)[0]
        M1 = L1 + Prior1 - Post1
        self.assertAlmostEqual(M1, P1.compute_marginal_likelihood()[0], 0)
        P0 = os.multivariate_stat(data[:, mask])
        P0.network[0] = 0
        P0.init_hidden_variables()
        P0.evaluate(nsimu=100, burnin=10, verbose=verbose)
        L0 = P0.compute_log_region_likelihood()[0]
        Prior0 = P0.compute_log_prior()[0]
        Post0 = P0.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)[0]
        M0 = L0 + Prior0 - Post0
        self.assertAlmostEqual(M0, P0.compute_marginal_likelihood()[0], 0)
        self.assertTrue(M1 > M0)
    
    def test_model_selection_mfx_spatial_rand_walk(self):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=3, noise=1, jitter=1, activation=True)
        #verbose=True
        P = os.multivariate_stat(data[:, mask], vardata[:, mask], XYZ[:, mask], std=1.0, sigma=5)
        P.network[0] = 0
        P.init_hidden_variables()
        P.evaluate(nsimu=100, burnin=100, verbose=verbose, proposal='rand_walk', proposal_std=1.0)
        #P.evaluate(nsimu=100, burnin=10, verbose=verbose, proposal='prior')
        L00 = P.compute_log_region_likelihood()[0]
        # Test simulated annealing procedure
        P.estimate_displacements_SA(nsimu=10, c=0.99, proposal_std=0.5, verbose=verbose)
        L0 = P.compute_log_region_likelihood()[0]
        self.assertTrue(L0 > L00)
        Prior0 = P.compute_log_prior()[0]
        Post0 = P.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)[0]
        M0 = L0 + Prior0 - Post0
        self.assertAlmostEqual(M0, P.compute_marginal_likelihood(verbose=verbose)[0], 0)
        #P = os.multivariate_stat(data[:, mask], vardata[:, mask], XYZ[:, mask], std=1, sigma=3)
        P.network[0] = 1
        P.init_hidden_variables(init_spatial=False)
        P.evaluate(nsimu=100, burnin=100, verbose=verbose, update_spatial=False)
        #L10 = P.compute_log_region_likelihood()[0]
        #P.estimate_displacements_SA(nsimu=10, c=0.99, proposal_std=0.5, verbose=verbose)
        L1 = P.compute_log_region_likelihood()[0]
        self.assertTrue(L1 > L10)
        Prior1 = P.compute_log_prior()[0]
        Post1 = P.compute_log_posterior(nsimu=1e2, burnin=1e2, verbose=verbose)[0]
        M1 = L1 + Prior1 - Post1
        self.assertAlmostEqual(M1, P.compute_marginal_likelihood(verbose=verbose)[0], 0)
        self.assertTrue(M1 > M0)


class test_multivariate_stat_mcmc(unittest.TestCase):
    
    def test_evaluate_mfx_spatial(self):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=False)
        #verbose=True
        P = os.multivariate_stat(data[:, mask], vardata[:, mask], XYZ[:, mask], std=1, sigma=3)
        P.init_hidden_variables()
        P.evaluate(nsimu=10, burnin=10, J = [P.D.XYZ_vol[10,10,10]],verbose=verbose, mode='mcmc')
        # Test log_likelihood computation
        v = P.v.copy()
        m_var = P.m_var.copy()
        m_mean = P.m_mean.copy()
        L1 = P.compute_log_region_likelihood_slow(v, m_mean, m_var)
        L2 = P.compute_log_region_likelihood(v, m_mean, m_var)
        self.assertAlmostEqual(-L1.sum(), L2.sum()*2, 2)
        # Test posterior density computation
        Prior = P.compute_log_prior(v, m_mean, m_var)
        Post = P.compute_log_posterior(v, m_mean, m_var, nsimu=10, burnin=10, verbose=verbose)

class test_labels_prior(unittest.TestCase):
     
    def test_update_labels(self):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        P = os.multivariate_stat(data[:, mask], vardata[:, mask], XYZ[:, mask])
        P.init_hidden_variables()
        p = P.data.shape[1]
        P.labels_prior = np.ones((1, p), float)
        P.label_values = np.zeros((1, p), int)
        P.labels_prior_mask = np.arange(p)
        P.update_labels()
        self.assertEqual(max(abs(P.labels - np.zeros(p, int))), 0)

if __name__ == "__main__":
    unittest.main()
