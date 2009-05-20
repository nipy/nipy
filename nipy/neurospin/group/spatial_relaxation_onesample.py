#####################################################################################
# BAYESIAN MODEL SELECTION FOR ACTIVATION DETECTION ON FMRI GROUP DATA
# Merlin Keller, 2009

import numpy as np
import scipy.special as sp

# Do inter-package imports the right way (this will become mandatory in py2.6)
# Alexis: I'm disabling this because this is incompatible with python
# 2.4 (needed for brainvisa)
"""
from ..group.routines import add_lines
from .displacement_field import displacement_field
"""
from routines import add_lines
from displacement_field import displacement_field

#####################################################################################
# some useful functions

def log_gammainv_pdf(x, a, b):
    """
    log density of the inverse gamma distribution with shape a and scale b,
    at point x, using Stirling's approximation for a > 100
    """
    L = a * np.log(b) - (a + 1) * np.log(x) - b / x
    if a <= 100:
        L -= np.log(sp.gamma(a))
    else:
        n = a - 1
        L -= 0.5 * np.log(2 * np.pi * n) + n * (np.log(n) - 1)
    return L

def log_gaussian_pdf(x, m, v):
    """
    log density of the gaussian distribution with mean m and variance v at point x
    """
    return -0.5 * (np.log(2 * np.pi * v) + (x - m)**2 / v)

#####################################################################################
# Main class

class multivariate_stat:
    def __init__(self, data, vardata=None, XYZ=None, std=None, sigma=None, 
                    labels=None, network=None, v_shape=1e-3, v_scale=1e-3, 
                    std_shape=1e-3, std_scale=1e-3, m_mean_rate=1e-3, 
                    m_var_shape=1e-3, m_var_scale=1e-3, disp_mask=None, 
                    labels_prior=None, label_values=None, labels_prior_mask=None):
        """
        Multivariate modeling of fMRI group data accounting for spatial uncertainty
        In: data         (n,p)        estimated effects
            vardata      (n,p)        variances of estimated effects
            XYZ          (3,p)        voxel coordinates
            std          <float>      Initial guess for standard deviate of spatial displacements
            sigma        <float>      regularity of displacement field
            labels       (p,)         labels defining regions of interest
            network      (N,)         binary region labels (1 for active, 0 for inactive)
            v_shape      <float>      intensity variance prior shape
            v_scale      <float>      intensity variance prior scale
            std_shape    <float>      spatial standard error prior shape
            std_scale    <float>      spatial standard error prior scale
            m_mean_rate  <float>      mean effect prior rate
            m_var_shape  <float>      effect variance prior shape
            m_var_scale  <float>      effect variance prior scale
            disp_mask    (q,)         mask of the brain, to limit displacements
            labels_prior (M,r)        prior on voxelwise region membership
            labels_prior_values (M,r) voxelwise label values where prior is defined
            labels_prior_mask (r,)    Mask of voxels where a label prior is defined
        """
        self.data = data
        if vardata != None and vardata.max() == 0:
            self.vardata = None
        else:
            self.vardata = vardata
        self.std = std
        self.sigma = sigma
        self.labels = labels
        self.network = network
        self.v_shape = v_shape
        self.v_scale = v_scale
        self.std_shape = std_shape
        self.std_scale = std_scale
        n, p = data.shape
        if labels == None:
            self.labels = np.zeros(p, int)
        M = self.labels.max() + 1
        if network == None:
            self.network = np.ones(M, int)
        if np.isscalar(m_mean_rate):
            self.m_mean_rate = np.zeros(M, float) + m_mean_rate
        else:
            self.m_mean_rate = m_mean_rate
        if np.isscalar(m_mean_rate):
            self.m_var_shape = np.zeros(M, float) + m_var_shape
        else:
            self.m_var_shape = m_var_shape
        if np.isscalar(m_mean_rate):
            self.m_var_scale = np.zeros(M, float) + m_var_scale
        else:
            self.m_var_scale = m_var_scale
        if std != None:
            self.D = displacement_field(XYZ, sigma, data.shape[0], disp_mask)
        self.labels_prior = labels_prior
        self.label_values = label_values
        self.labels_prior_mask = labels_prior_mask
    
    def init_hidden_variables(self, mode='saem', init_spatial=True):
        n, p = self.data.shape
        self.X = self.data.copy()
        self.m = self.X.mean(axis=0)
        #self.v = np.square(self.X - self.m).mean()
        N = len(self.network)
        self.m_mean = np.zeros(N, float)
        self.m_var = np.zeros(N, float)
        self.v = np.zeros(N, float)
        #self.s0 = np.zeros(N, float)
        #self.S0 = np.zeros(N, float)
        self.s1 = np.zeros(N, float)
        self.S1 = np.zeros(N, float)
        self.s2 = np.zeros(N, float)
        self.S2 = np.zeros(N, float)
        self.s3 = np.zeros(N, float)
        self.S3 = np.zeros(N, float)
        if self.labels_prior != None:
            self.s6 = np.zeros(N, float)
        self.region_size = np.zeros(N, float)
        for j in xrange(N):
            self.region_size[j] = (self.labels == j).sum()
        self.m_var_post_scale = np.zeros(N, float)
        if init_spatial and self.std != None:
            B = len(self.D.block)
            if B == 0:
                self.std = None
            else:
                self.R = np.zeros((n, B), int)
                self.N = np.ones(p, float) * n
                self.s4 = 0.0
                self.S4 = 0.0
                self.s5 = np.zeros(N, float)
                self.S5 = np.zeros(N, float)
        std = self.std
        self.update_summary_statistics(init_spatial)
        if mode == 'saem':
            self.update_parameters_saem(init_spatial)
        else:
            self.update_parameters_mcmc(init_spatial)
        self.std = std
    
    def update_summary_statistics(self, w=1.0, update_spatial=True):
        n, p = self.data.shape
        if self.std == None:
            m = self.m
        else:
            m = self.m[self.D.I]
            if update_spatial:
                self.s4 = (self.D.U**2).sum()
                self.S4 += w * (self.s4 - self.S4)
        if self.vardata == None:
            SS = (self.data - m)**2 #/ self.v + np.log(2 * np.pi * self.v)
        else:
            SS = (self.X - self.m)**2 #/ self.vardata + np.log(2 * np.pi * self.vardata)
        if self.std == None:
            SS_sum = SS.sum(axis=0)
        else:
            SS_sum = np.zeros(p, float)
            for i in xrange(n):
                Ii = self.D.I[i]
                SSi = SS[i].reshape(p, 1)
                add_lines(SSi, SS_sum.reshape(p, 1), Ii)
        #self.s1 = ((self.X - m)**2).sum()
        for j in xrange(len(self.network)):
            L = np.where(self.labels == j)[0]
            self.s1[j] = SS_sum[L].sum()
            if self.labels_prior != None:
                self.s6[j] = len(L)
            #self.s0[j] = SS_sum[L].sum()
            self.s2[j] = (self.m[L]**2).sum()
            if self.network[j] == 1:
                self.s3[j] = self.m[L].sum()
            if update_spatial and self.std != None:
                self.s5[j] = self.N[L].sum()
                self.S5 += w * (self.s5 - self.S5)
        #self.S0 += w * (self.s0 - self.S0)
        self.S1 += w * (self.s1 - self.S1)
        self.S2 += w * (self.s2 - self.S2)
        self.S3 += w * (self.s3 - self.S3)
        if self.labels_prior != None:
            self.region_size += w * (self.s6 - self.region_size)
    
    def update_parameters_saem(self, update_spatial=True):
        n, p = self.data.shape
        #self.v = (self.S1 + 2 * self.v_scale) / (n * p + 2 * (1 + self.v_shape))
        for j in xrange(len(self.network)):
            rate = self.m_mean_rate[j]
            shape = self.m_var_shape[j]
            scale = self.m_var_scale[j]
            pj = self.region_size[j]
            if self.std == None:
                Nj = n * pj
            else:
                Nj = self.S5[j]
            self.v[j] = (self.S1[j] + 2 * self.v_scale) / (Nj + 2 * (1 + self.v_shape))
            if self.network[j] == 1:
                self.m_mean[j] = self.S3[j] / (pj + rate)
                self.m_var_post_scale[j] = scale + 0.5 * (self.S2[j] - self.S3[j]**2 / (pj + rate))
                self.m_var[j] = 2 * self.m_var_post_scale[j] / (pj + 2 * shape + 3)
            else:
                self.m_var_post_scale[j] = scale + 0.5 * self.S2[j]
                self.m_var[j] = 2 * self.m_var_post_scale[j] / (pj + 2 * shape + 2)
        if update_spatial and self.std != None:
            B = len(self.D.block)
            self.std = np.sqrt((self.S4 + 2 * self.std_scale) / (3 * n * B + 2 * self.std_shape + 1))
    
    def update_parameters_mcmc(self, update_spatial=True):
        n, p = self.data.shape
        #self.v = (self.s1 + 2 * self.v_scale) / np.random.chisquare(df = n * p + 2 * self.v_shape)
        for j in xrange(len(self.network)):
            rate = self.m_mean_rate[j]
            shape = self.m_var_shape[j]
            scale = self.m_var_scale[j]
            if self.labels_prior == None:
                pj = self.region_size[j]
            else:
                pj = self.s6[j]
            if self.std == None:
                Nj = n * pj
            else:
                Nj = self.s5[j]
            self.v[j] = (self.s1[j] + 2 * self.v_scale) / np.random.chisquare(df = Nj + 2 * self.v_shape)
            if self.network[j] == 1:
                s3 = self.s3[j]
                post_rate = rate + pj
                self.m_var_post_scale[j] = scale + 0.5 * (self.s2[j] - s3**2 / post_rate)
                self.m_var[j] = 2 * self.m_var_post_scale[j] / np.random.chisquare(df = pj + 2 * shape)
                self.m_mean[j] = s3 / post_rate + np.random.randn() * np.sqrt(self.m_var[j] / post_rate)
            else:
                self.m_var_post_scale[j] = scale + 0.5 * self.s2[j]
                self.m_var[j] = 2 * self.m_var_post_scale[j] / np.random.chisquare(df = pj + 2 * shape)
        if update_spatial and self.std != None:
            B = len(self.D.block)
            self.std = np.sqrt((self.s4 + 2 * self.std_scale) / np.random.chisquare(df = 3 * n * B + 2 * self.std_shape))
    
    def update_displacements(self):
        n = self.data.shape[0]
        B = len(self.D.block)
        if self.proposal == 'prior':
            for i in xrange(n):
                for b in np.random.permutation(range(B)):
                    block = self.D.block[b]
                    self.update_block(i, b, 'prior', self.std)
        elif self.proposal == 'rand_walk':
            if np.isscalar(self.proposal_std):
                for i in xrange(n):
                    for b in np.random.permutation(range(B)):
                        block = self.D.block[b]
                        self.update_block(i, b, 'rand_walk', self.proposal_std * self.std)
            else:
                for i in xrange(n):
                    for b in np.random.permutation(range(B)):
                        block = self.D.block[b]
                        self.update_block(i, b, 'rand_walk', self.proposal_std[:, i, b])
        else:
            for i in xrange(n):
                for b in np.random.permutation(range(B)):
                    block = self.D.block[b]
                    self.update_block(i, b, 'fixed', self.proposal_std[:, i, b], self.proposal_mean[:, i, b])
        if self.verbose:
            print "mean rejected displacements :", self.R.mean(axis=0)
    
    def update_block(self, i, b, proposal='prior', proposal_std=None, 
                            proposal_mean=None, verbose=False):
        block = self.D.block[b]
        if verbose:
            print 'sampling field', i, 'block', b
        # Propose new displacement
        U, V, L, W, I = self.D.sample(i, b, proposal, proposal_std, 
                                                        proposal_mean)
        Uc = self.D.U[:, i, b]
        Ic = self.D.I[i, L]
        # log acceptance rate
        mc = self.m[Ic]
        m = self.m[I]
        vc = self.v[self.labels[Ic]]
        v = self.v[self.labels[I]]
        #A = ((mc - m) * (mc + m - 2 * self.X[i, L])).sum() / self.v
        A = (np.log(v) - np.log(vc)
            + (self.X[i, L] - mc)**2 / vc
            - (self.X[i, L] - m)**2 / v).sum()
        if not proposal == 'prior':
            A += (Uc**2 - U**2).sum() / self.std**2
            if proposal == 'fixed':
                A += ((U - Uc) * (U + Uc - 2 * proposal_mean) / proposal_std**2).sum()
        self.R[i, b] = np.random.uniform() > np.exp(0.5 * A)
        if self.R[i, b] == 0:
            self.D.U[:, i, b] = U
            self.D.V[:, i, block] = V
            if len(L)> 0:
                self.D.W[:, i, L] = W
                self.D.I[i, L] = I
    
    def update_effects(self):
        n, p = self.data.shape
        if self.std == None:
            m = self.m
            v = self.v[self.labels]
        else:
            m = self.m[self.D.I]
            v = self.v[self.labels[self.D.I]]
        #tot_var = self.v + self.vardata
        #cond_mean = (self.v * self.data + self.vardata * m) / tot_var
        #cond_var = self.v * self.vardata / tot_var
        tot_var = v + self.vardata
        cond_mean = (v * self.data + self.vardata * m) / tot_var
        cond_var = v * self.vardata / tot_var
        self.X = cond_mean + np.random.randn(n, p) * np.sqrt(cond_var)
    
    def update_mean_effect(self):
        n, p = self.data.shape
        X_sum = np.zeros(p, float)
        if self.std == None:
            X_sum = self.X.sum(axis=0)
        else:
            self.N *= 0
            ones = np.ones((p, 1), float)
            for i in xrange(n):
                Ii = self.D.I[i]
                XI = self.X[i].reshape(p, 1)
                add_lines(XI, X_sum.reshape(p, 1), Ii)
                add_lines(ones, self.N.reshape(p, 1), Ii)
        for j in xrange(len(self.network)):
            L = np.where(self.labels == j)[0]
            m_var = self.m_var[j]
            v = self.v[j]
            if self.std == None:
                #tot_var = self.v + m_var * n
                tot_var = v + m_var * n
            else:
                #tot_var = self.v + m_var * self.N[L]
                tot_var = v + m_var * self.N[L]
            #cond_mean = (X_sum[L] * m_var + self.v * self.m_mean[j]) / tot_var
            #cond_std = np.sqrt(self.v * m_var / tot_var)
            cond_mean = (X_sum[L] * m_var + v * self.m_mean[j]) / tot_var
            cond_std = np.sqrt(v * m_var / tot_var)
            self.m[L] = cond_mean + np.random.randn(len(L)) * cond_std
    
    def update_labels(self):
        N, r = self.labels_prior.shape
        I = self.labels_prior_mask
        m_mean = self.m_mean[self.label_values]
        m_var = self.m_var[self.label_values]
        L = (self.m[I].reshape(1, r) - m_mean)**2 / m_var
        P = self.labels_prior * np.exp(-0.5 * L) / np.sqrt(m_var)
        P_cumsum = P.cumsum(axis=0)
        X = np.random.rand(r) * P_cumsum[-1]
        labels = (X > P_cumsum).sum(axis=0)
        self.labels[I] = self.label_values[labels, xrange(r)]
    
    def evaluate(self, nsimu=1e3, burnin=1e2, J=None, verbose=False, 
                    proposal='prior', proposal_std=None, proposal_mean=None, 
                    compute_post_mean=False, mode='saem', update_spatial=True):
        """
        Sample posterior distribution of model parameters, or compute their MAP estimator
        In:  nsimu            <int>             Number of samples drawn from posterior mean distribution
             burnin           <int>             Number of discarded burn-in samples
             J                (N,)              voxel indices where successive mean values are stored
             verbose          <bool>            Print some infos during the sampling process
             proposal         <str>             'prior', 'rand_walk' or 'fixed'
             proposal_mean    <float>           Used for fixed proposal only
             proposal_std     <float>           Used for random walk or fixed proposal
             mode             <str>             if mode='saem', compute MAP estimates of model parameters.
                                               if mode='mcmc', sample their posterior distribution
             update_spatial   <bool>            when False, enables sampling conditional on spatial parameters
        Out: self.m_values    (N, nsimu+burnin) successive mean values (if J is not empty)
        if self.labels_prior is not empty:
             self.labels_post (M,r)             posterior distribution of region labels
        if self.std is not empty:
             self.std_values  (nsimu+burnin,)   successive spatial standard deviate values
        if compute_post_mean is True:
             self.mean_m      (p,)              posterior average of mean effect
             self.var_m       (p,)              posterior variance of mean effect
        if self.std is not empty and compute_post_mean is True:
             self.r           (n, nblocks)      mean rejection rate for each displacement field
             self.mean_U      (3, n, nblocks)   posterior average of displacement weights
             self.var_U       (3, n, nblocks)   posterior marginal variances of displacement weights
        """
        #self.init_hidden_variables()
        n, p = self.data.shape
        self.nsimu = nsimu
        self.burnin = burnin
        self.J = J
        self.verbose = verbose
        self.proposal = proposal
        self.proposal_mean = proposal_mean
        self.proposal_std = proposal_std
        self.compute_post_mean = compute_post_mean
        #self.v_values = np.zeros(nsimu + burnin, float)
        if J != None:
            self.m_values = np.zeros((len(J), nsimu + burnin), float)
        if self.std != None:
            B = len(self.D.block)
            if update_spatial:
                self.std_values = self.std_values = np.zeros(nsimu + burnin, float)
        if self.labels_prior != None:
            self.labels_post = np.zeros(self.labels_prior.shape, float)
            #Il = np.array(np.where(self.labels_prior > 0))
            #r = len(self.labels_prior_mask)
        if compute_post_mean:
            sum_m = np.zeros(p, float)
            sum_m_sq = np.zeros(p, float)
            if mode == 'mcmc':
                self.P = np.zeros(len(self.network), float)
            if update_spatial and self.std != None:
                self.r = np.zeros((n, B), float)
                sum_U = np.zeros((3, n, B), float)
                sum_U_sq = np.zeros((3, n, B), float)
        niter = np.array([int(burnin), int(nsimu)])
        for j in np.arange(2)[niter>0]:
            if j == 0:
                w = 1
                if self.verbose:
                    print "Burn-in"
            else:
                if mode == 'saem':
                    if self.verbose:
                        print "Maximizing likelihood"
                else:
                    if self.verbose:
                        print "Sampling posterior distribution"
            for i in xrange(niter[j]):
                if self.verbose:
                    if mode == 'saem':
                        print "SAEM",
                    else:
                        print "Gibbs",
                    print "iteration", i+1, "out of", niter[j]
                # Gibbs iteration
                #i += 1
                if update_spatial and self.std != None:
                    self.update_displacements()
                if self.vardata != None:
                    self.update_effects()
                self.update_mean_effect()
                if self.labels_prior != None:
                    self.update_labels()
                if j == 1:
                    w = 1.0 / (i + 1)
                self.update_summary_statistics(w, update_spatial)
                if mode == 'saem':
                    self.update_parameters_saem(update_spatial)
                else:
                    self.update_parameters_mcmc(update_spatial)
                if self.verbose:
                    print "population effect min variance value :", self.m_var.min()
                # Update results
                #self.v_values[i + self.burnin * j] = self.v
                if update_spatial and self.std != None:
                    self.std_values[i + self.burnin * j] = self.std
                if self.J != None:
                    self.m_values[:, i + self.burnin * j] = self.m[self.J]
                if j == 1 and self.labels_prior != None:
                    self.labels_post += self.label_values == self.labels[self.labels_prior_mask]
                    #self.labels_post[Il[0], Il[1]] += self.label_values[Il[0], Il[1]] == self.labels[Il[0]]
                if j == 1 and compute_post_mean:
                    sum_m += self.m
                    sum_m_sq += self.m**2
                    if mode == 'mcmc':
                        self.P += self.m_mean > 0
                    if update_spatial and self.std != None:
                        self.r += self.R
                        sum_U += self.D.U
                        sum_U_sq += self.D.U**2
            if j== 1 and self.labels_prior != None:
                self.labels_post /= nsimu
            if j == 1 and compute_post_mean:
                self.mean_m = sum_m / float(self.nsimu)
                self.var_m = sum_m_sq / float(self.nsimu) - self.mean_m**2
                if mode == 'mcmc':
                    self.P /= float(self.nsimu)
                if update_spatial and self.std != None:
                    self.r /= float(self.nsimu)
                    self.mean_U = sum_U / float(self.nsimu)
                    self.var_U = sum_U_sq / float(self.nsimu) - self.mean_U**2
    
    #####################################################################################
    # MAP estimation of displacement fields
    
    def estimate_displacements_SA(self, nsimu=1e2, c=0.99, proposal_std=1.0, verbose=False):
        """
        MAP estimate of elementary displacements conditional on model parameters
        """
        for i in xrange(nsimu):
            if verbose:
                print "SA iteration", i+1, "out of", nsimu
            self.update_displacements_SA(c**i, proposal_std, verbose)
            self.update_summary_statistics(w=1.0, update_spatial=True)
    
    def update_displacements_SA(self, T=1.0, proposal_std=None, verbose=False):
        n = self.data.shape[0]
        B = len(self.D.block)
        for i in xrange(n):
            for b in np.random.permutation(range(B)):
                block = self.D.block[b]
                self.update_block_SA(i, b, T, proposal_std * self.std, verbose)
        if self.verbose:
            print "mean rejected displacements :", self.R.mean(axis=0)
    
    def update_block_SA(self, i, b, T=1.0, proposal_std=None, verbose=False):
        """
        Update displacement block using simulated annealing scheme 
        with random-walk kernel
        """
        if proposal_std==None:
            proposal_std=self/std
        block = self.D.block[b]
        if verbose:
            print 'sampling field', i, 'block', b
        # Propose new displacement
        U, V, L, W, I = self.D.sample(i, b, 'rand_walk', proposal_std * T)
        Uc = self.D.U[:, i, b]
        Vc = self.D.V[:, i, block]
        Wc = self.D.W[:, i, L]
        Ic = self.D.I[i, L]
        # log acceptance rate
        fc = self.compute_log_voxel_likelihood().sum()
        self.D.U[:, i, b] = U
        self.D.V[:, i, block] = V
        if len(L)> 0:
            self.D.W[:, i, L] = W
            self.D.I[i, L] = I
        f = self.compute_log_voxel_likelihood().sum()
        A = (f - fc + (Uc**2 - U**2).sum() / self.std**2)
        self.R[i, b] = np.random.uniform() > np.exp(A / T)
        if self.R[i, b] == 1:
            self.D.U[:, i, b] = Uc
            self.D.V[:, i, block] = Vc
            if len(L)> 0:
                self.D.W[:, i, L] = Wc
                self.D.I[i, L] = Ic
    
    #####################################################################################
    # Marginal likelihood computation for model selection
    
    def compute_log_region_likelihood_slow(self, v=None, m_mean=None, m_var=None, verbose=False, J=None):
        """
        Essentially maintained for debug purposes
        """
        if v == None:
            v = self.v
        if m_mean == None:
            m_mean = self.m_mean
        if m_var == None:
            m_var = self.m_var
        n, p = self.data.shape
        nregions = len(self.network)
        log_region_likelihood = np.zeros(nregions, float)
        if J == None:
            J = xrange(nregions)
        if self.std == None:
            nk = n
        else:
            I = self.D.I
            argsort_I = np.argsort(I.ravel())
            data_I = self.data.ravel()[argsort_I]
            if self.vardata != None:
                var_I = (self.vardata + v[self.labels[I]]).ravel()[argsort_I]
            cumsum = np.zeros(p + 1, int)
            cumsum[1:] = self.N.cumsum().astype(int)
        for i in xrange(len(J)):
            j = J[i]
            if verbose:
                print "computing log likelihood for region", i + 1, "out of", len(J)
            m_var_j = self.m_var[j]
            m_mean_j = self.m_mean[j]
            v_j = self.v[j]
            L = np.where(self.labels == j)[0]
            for k in L:
                if self.std == None:
                    datak = np.matrix(self.data[:, k].reshape(n, 1) - m_mean_j)
                    if self.vardata != None:
                        vark = self.vardata[:, k] + v_j
                else:
                    nk = int(self.N[k])
                    datak = np.matrix(data_I[cumsum[k] : cumsum[k + 1]].reshape(nk, 1) - m_mean_j)
                    if self.vardata != None:
                        vark = var_I[cumsum[k] : cumsum[k + 1]]
                Vk = np.matrix(np.zeros((nk, nk), float) + m_var_j)
                if self.vardata == None:
                    Vk[xrange(nk), xrange(nk)] = v_j + m_var_j
                else:
                    Vk[xrange(nk), xrange(nk)] = vark + m_var_j
                log_region_likelihood[j] += np.log(np.linalg.det(Vk)) + datak.transpose() * np.linalg.inv(Vk) * datak
            if self.std == None:
                nj = n * len(L)
            else:
                nj = self.N[L].sum()
            log_region_likelihood[j] += nj * np.log(2 * np.pi)
            return log_region_likelihood
    
    def compute_log_region_likelihood(self, v=None, m_mean=None, m_var=None):
        log_voxel_likelihood = self.compute_log_voxel_likelihood(v, m_mean, m_var)
        N = len(self.network)
        log_region_likelihood = np.zeros(N, float)
        for j in xrange(N):
            log_region_likelihood[j] = log_voxel_likelihood[self.labels==j].sum()
        return log_region_likelihood
    
    def compute_log_voxel_likelihood(self, v=None, m_mean=None, m_var=None):
        if v == None:
            v = self.v
        if m_mean == None:
            m_mean = self.m_mean
        if m_var == None:
            m_var = self.m_var
        n, p = self.data.shape
        if self.std == None:
            N = n
            v_labels = v[self.labels]
            Z = self.data - m_mean[self.labels]
        else:
            N = self.N
            I = self.D.I
            v_labels = v[self.labels[I]]
            Z = self.data - m_mean[self.labels[I]]
        if self.vardata == None:
            tot_var = v_labels + np.zeros(self.data.shape, float)
        else:
            tot_var = v_labels + self.vardata
        if self.std == None:
            SS1 = (1 / tot_var).sum(axis=0)
            SS2 = np.log(tot_var).sum(axis=0)
            SS3 = (Z**2 / tot_var).sum(axis=0)
            SS4 = (Z / tot_var).sum(axis=0)
        else:
            SS1 = np.zeros(p, float)
            SS2 = np.zeros(p, float)
            SS3 = np.zeros(p, float)
            SS4 = np.zeros(p, float)
            for i in xrange(n):
                Ii = self.D.I[i]
                add_lines((1 / tot_var[i]).reshape(p, 1), SS1.reshape(p, 1), Ii)
                add_lines(np.log(tot_var[i]).reshape(p, 1), SS2.reshape(p, 1), Ii)
                add_lines((Z[i]**2 / tot_var[i]).reshape(p, 1), SS3.reshape(p, 1), Ii)
                add_lines((Z[i] / tot_var[i]).reshape(p, 1), SS4.reshape(p, 1), Ii)
        return - 0.5 * (\
            N * np.log(2 * np.pi) + np.log(1 + m_var * SS1) \
            + SS2 + SS3 - SS4**2 / (1 / m_var[self.labels] + SS1))
    
    def compute_log_prior(self, v=None, m_mean=None, m_var=None, std=None):
        """
        compute log prior density of model parameters, spatial uncertainty excepted,
        assuming hidden variables have been initialized
        """
        if v == None:
            v = self.v
        if m_mean == None:
            m_mean = self.m_mean
        if m_var == None:
            m_var = self.m_var
        if std == None:
            std = self.std
        N = len(self.network)
        log_prior_values = np.zeros(N + 1, float)
        for j in xrange(N):
            pj = self.region_size[j]
            log_prior_values[j] = log_gammainv_pdf(v[j], self.v_shape, self.v_scale)
            log_prior_values[j] += log_gammainv_pdf(m_var[j], self.m_var_shape[j], self.m_var_scale[j])
            if self.network[j] == 1:
                log_prior_values[j] += log_gaussian_pdf(m_mean[j], 0, m_var[j] / self.m_mean_rate[j])
        if self.std != None:
            log_prior_values[-1] = log_gammainv_pdf(std**2, self.std_shape, self.std_scale)
        return log_prior_values
    
    def compute_log_conditional_posterior(self, v=None, m_mean=None, m_var=None, std=None):
        """
        compute log posterior density of model parameters, conditional on hidden parameters.
        This function is used in compute_log_region_posterior. It should only be used with
        the Gibbs sampler, and not the SAEM algorithm.
        """
        n,p = self.data.shape
        if v == None:
            v = self.v
        if m_mean == None:
            m_mean = self.m_mean
        if m_var == None:
            m_var = self.m_var
        if std == None:
            std = self.std
        N = len(self.network)
        log_conditional_posterior = np.zeros(N + 1, float)
        for j in xrange(N):
            pj = self.region_size[j]
            if self.std == None:
                Nj = n * pj
            else:
                Nj = self.s5[j]
            log_conditional_posterior[j] = \
              log_gammainv_pdf(v[j], self.v_shape + 0.5 * Nj, self.v_scale + 0.5 * self.s1[j])
            log_conditional_posterior[j] += \
              log_gammainv_pdf(m_var[j], self.m_var_shape[j] + 0.5 * pj, self.m_var_post_scale[j])
            if self.network[j] == 1:
                post_rate = self.m_mean_rate[j] + pj
                log_conditional_posterior[j] += \
                  log_gaussian_pdf(m_mean[j], self.s3[j] / post_rate, m_var[j] / post_rate)
        n, p = self.data.shape
        if self.std != None:
            B = len(self.D.block)
            log_conditional_posterior[-1] = \
              log_gammainv_pdf(std**2, self.std_shape + 0.5 * 3 * n * B, self.std_scale + 0.5 * self.s4)
        return log_conditional_posterior
    
    def sample_log_conditional_posterior(self, v=None, m_mean=None, m_var=None, nsimu=1e2, burnin=1e2, verbose=False):
        """
        sample log conditional posterior density of region parameters
        using a Gibbs sampler (assuming all hidden variables have been initialized)
        """
        if v == None:
            v = self.v.copy()
        if m_mean == None:
            m_mean = self.m_mean.copy()
        if m_var == None:
            m_var = self.m_var.copy()
        N = len(self.network)
        log_conditional_posterior_values = np.zeros((nsimu, N), float)
        #self.init_hidden_variables()
        n, p = self.data.shape
        self.nsimu = nsimu
        self.burnin = burnin
        #self.J = J
        self.verbose = verbose
        niter = np.array([int(burnin), int(nsimu)])
        for j in np.arange(2)[niter>0]:
            if self.verbose:
                if j == 0:
                    print "Burn-in"
                else:
                    print "Sampling posterior distribution"
            for i in xrange(niter[j]):
                if self.verbose:
                    print "Iteration", i+1, "out of", niter[j]
                # Gibbs iteration
                #i += 1
                if self.vardata != None:
                    self.update_effects()
                self.update_mean_effect()
                #if self.labels_prior != None:
                    #self.update_labels()
                self.update_summary_statistics(update_spatial=False)
                #self.update_parameters_mcmc(update_spatial=False)
                if self.verbose:
                    print "population effect min variance value :", self.m_var.min()
                if j == 1:
                    log_conditional_posterior_values[i] = \
                    self.compute_log_conditional_posterior(v, m_mean, m_var)[:-1]
        return log_conditional_posterior_values
    
    def compute_log_posterior(self, v=None, m_mean=None, m_var=None, nsimu=1e2, burnin=1e2, verbose=False):
        """
        compute log posterior density of region parameters by Rao-Blackwell method
        using a Gibbs sampler (assuming all hidden variables have been initialized)
        """
        log_conditional_posterior_values \
            = self.sample_log_conditional_posterior(v, m_mean, m_var, nsimu, burnin, verbose)
        max_log_conditional = log_conditional_posterior_values.max(axis=0)
        ll_ratio = log_conditional_posterior_values - max_log_conditional
        return max_log_conditional + np.log(np.exp(ll_ratio).sum(axis=0)) - np.log(nsimu)
        #return max_log_conditional #+ ll_ratio.mean(axis=0)
    
    def compute_marginal_likelihood(self, v=None, m_mean=None, m_var=None, nsimu=1e2, burnin=1e2, verbose=False):
        log_likelihood = self.compute_log_region_likelihood(v, m_mean, m_var)
        log_prior = self.compute_log_prior(v, m_mean, m_var)
        log_posterior = self.compute_log_posterior(v, m_mean, m_var, nsimu, burnin, verbose)
        return log_likelihood + log_prior[:-1] - log_posterior

