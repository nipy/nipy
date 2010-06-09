# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np 
import pylab 
import os

from _mrf import _ve_step, _concensus

TINY = 1e-30 


# VM-step 
def gauss_dist(x, mu, sigma): 
    return np.exp(-.5*((x-mu)/sigma)**2)/sigma

def laplace_dist(x, mu, sigma): 
    return np.exp(-np.abs((x-mu)/sigma))/sigma

def vm_step_gauss(ppm, data_, mask): 
    """
    ppm: ndarray (4d)
    data_: ndarray (1d, masked data)
    mask: 3-element tuple of 1d ndarrays (X,Y,Z)
    """
    ntissues = ppm.shape[3]
    mu = np.zeros(ntissues)
    sigma = np.zeros(ntissues)

    for i in range(ntissues):
        P = ppm[:,:,:,i][mask]
        Z = P.sum()
        tmp = data_*P
        mu_ = tmp.sum()/Z
        sigma_ = np.sqrt(np.sum(tmp*data_)/Z - mu_**2)
        mu[i] = mu_ 
        sigma[i] = sigma_
    return mu, sigma 


def wmedian(x, w, ind): 
    F = np.cumsum(w[ind])
    f = .5*(w.sum()+1)
    i = np.searchsorted(F, f)
    if i == 0: 
        return x[ind[0]]
    wr = (f-F[i-1])/(F[i]-F[i-1])
    jr = ind[i]
    jl = ind[i-1]
    return wr*x[jr]+(1-wr)*x[jl]

def vm_step_laplace(ppm, data_, mask): 
    """
    ppm: ndarray (4d)
    data_: ndarray (1d, masked data)
    mask: 3-element tuple of 1d ndarrays (X,Y,Z)
    """
    ntissues = ppm.shape[3]
    mu = np.zeros(ntissues)
    sigma = np.zeros(ntissues)
    ind = np.argsort(data_) # data_[ind] increasing

    for i in range(ntissues):
        P = ppm[:,:,:,i][mask]
        mu_ = wmedian(data_, P, ind) 
        sigma_ = np.sum(np.abs(P*(data_-mu_)))/P.sum()
        mu[i] = mu_ 
        sigma[i] = sigma_
    return mu, sigma 



# VEM algorithm 
class VemTissueClassification(object): 

    def __init__(self, ppm, data, mask, noise='gauss', 
                 prior=True, copy=False, hard=False, 
                 labels=None, mixmat=None): 
        """
        data: ndarray (3d)
        mask: 3-element tuple of 1d ndarrays (X,Y,Z)
        
        output: 
        ppm: ndarray (4d)
        """
        self.copy = copy
        self.hard = hard
        if noise == 'gauss': 
            self.dist = gauss_dist
            self._vm_step = vm_step_gauss
        elif noise == 'laplace':
            self.dist = laplace_dist
            self._vm_step = vm_step_laplace
        else:
            raise ValueError('Unknown noise model')

        # Mask data 
        self.ppm = ppm 
        self.ntissues = ppm.shape[3]
        self.mask = mask 
        self.data_ = data[mask]
        if prior: 
            self.prior_ = ppm[mask]
        else: 
            self.prior_ = np.ones([1,self.ntissues])/float(self.ntissues)
        self.ref_ = np.zeros([self.data_.size, self.ntissues])

        # Label information 
        if labels == None: 
            labels = [str(l) for l in range(self.ntissues)]
        if not len(labels) == self.ntissues: 
            raise ValueError('Wrong length for labels sequence') 
        self.labels = labels

        # Mixing matrix 
        self.mixmat = mixmat


    # VM-step: estimate parameters
    def vm_step(self): 
        """
        Return (mu, sigma)
        """
        return self._vm_step(self.ppm, self.data_, self.mask)

    def sort_labels(self, mu):
        K = len(mu)
        tmp = np.asarray(self.labels)
        labels = np.zeros(K, dtype=tmp.dtype)
        labels[np.argsort(mu)] = tmp
        return list(labels)

    def sort_mixmat(self, mu): 
        K = len(mu)
        mixmat = np.zeros([K,K]) 
        I, J = np.mgrid[0:K, 0:K]
        idx = np.argsort(mu) 
        mixmat[idx[I], idx[J]] = np.asarray(self.mixmat)
        return mixmat
    


    # VE-step: update tissue probability map
    def ve_step(self, mu, sigma, alpha=1., beta=0.0): 
        """
        VE-step
        """
        for i in range(self.ntissues): 
            self.ref_[:,i] = self.prior_[:,i]**alpha
            self.ref_[:,i] *= self.dist(self.data_, mu[i], sigma[i])

        # Replace very small values for numerical stability 
        self.ref_[:] = np.maximum(self.ref_, TINY) 
        
        # Normalize reference probability map 
        if beta == 0.0: 
            self.ppm[self.mask] = (self.ref_.T/self.ref_.sum(1)).T

        # Update and normalize reference probabibility map using
        # neighborhood information (mean-field theory)
        else: 
            print('  ... MRF correction')
            # Deal with mixing matrix and label switching
            if not self.mixmat == None:
                mixmat = self.sort_mixmat(mu)
            self.ppm = _ve_step(self.ppm, self.ref_, 
                                np.array(self.mask, dtype='int'), 
                                beta, self.copy, self.hard, mixmat)
            

    def __call__(self, mu=None, sigma=None, alphas=None, betas=None, niters=5): 

        if betas == None: 
            betas = np.zeros(niters)
        else: 
            niters = len(betas)
        if alphas == None: 
            alphas = np.ones(niters)

        if not len(alphas) == niters: 
            raise ValueError('Inconsistent length for alphas and betas.')

        do_vm_step = (mu==None)
        if not do_vm_step: 
            mu = np.asarray(mu, dtype='double')
            sigma = np.asarray(sigma, dtype='double')
        
        for i in range(niters):
            print('VEM iter %d/%d' % (i+1, niters))
            print('  VM-step...')
            if do_vm_step: 
                mu, sigma = self.vm_step()
            print('  VE-step...')
            self.ve_step(mu, sigma, alpha=alphas[i], beta=betas[i])
            do_vm_step = True

        return mu, sigma 


    def free_energy(self, beta=0.0):
        q_ = self.ppm[self.mask]
        f = np.sum(q_*np.log(np.maximum(q_/self.ref_, TINY)))
        if beta > 0.0: 
            print('  ... Concensus correction')
            fc = _concensus(self.ppm, np.array(self.mask, dtype='int'))
            print fc
            f = f - .5*beta*fc 
        return f
