import unittest

import numpy as np
import scipy.stats as st

import neuroimaging.neurospin.group.spatial_relaxation_onesample as os

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
        XYZvol *= 0
        XYZvol[XYZ[0,mask],XYZ[1,mask],XYZ[2,mask]] = np.arange(len(mask))
        ##P = os.stat(data, XYZ, mask, mean_prior_rate = 0, var_prior_scale=0, var_prior_shape=-0.5)
        P = os.multivariate_stat(data[:, mask])
        P.init_hidden_variables()
        P.evaluate(nsimu=100, burnin=100, J = [XYZvol[5,5,5]], compute_post_mean=True)
        P.compute_log_likelihood_regionwise()
        # Verify code consistency
        Q = os.multivariate_stat(data[:, mask], vardata[:, mask] * 0, XYZ[:, mask], std=0, sigma=5)
        Q.init_hidden_variables()
        Q.evaluate(nsimu=100, burnin=100, J = [XYZvol[5,5,5]], compute_post_mean=True, update_spatial=False)
        Q.compute_log_likelihood_regionwise()
        self.assertAlmostEqual(P.mean_m.mean(), Q.mean_m.mean(), int(np.log10(P.nsimu))-1)
        self.assertAlmostEqual(Q.log_likelihood_values.sum(), P.log_likelihood_values.sum(), 2)
    
    def test_evaluate_mfx_spatial_rand_walk(self):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        P = os.multivariate_stat(data[:, mask], vardata[:, mask], XYZ[:, mask], std=1, sigma=3)
        P.init_hidden_variables()
        P.evaluate(nsimu=10, burnin=10, J = [P.D.XYZ_vol[10,10,10]],verbose=False, proposal='rand_walk', proposal_std = 0.01)
    
    def test_evaluate_mfx_spatial_fixed(self):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        P = os.multivariate_stat(data[:, mask], vardata[:, mask], XYZ[:, mask], std=1, sigma=3)
        P.init_hidden_variables()
        P.evaluate(nsimu=10, burnin=10, J = [P.D.XYZ_vol[10,10,10]],verbose=False, compute_post_mean=True)
        proposal_mean = P.mean_U
        proposal_std = np.sqrt(np.clip(P.var_U, 1e-4, 4))
        P.evaluate(10, 0, [P.D.XYZ_vol[10,10,10]], False, 'fixed', proposal_std, proposal_mean)


class test_multivariate_stat_mcmc(unittest.TestCase):
    
    def test_evaluate_mfx_spatial(self):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        P = os.multivariate_stat(data[:, mask], vardata[:, mask], XYZ[:, mask], std=1, sigma=3)
        P.init_hidden_variables()
        P.evaluate(nsimu=10, burnin=10, J = [P.D.XYZ_vol[10,10,10]],verbose=False, mode='mcmc')

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
