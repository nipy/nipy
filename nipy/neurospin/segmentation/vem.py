# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np 
import pylab 
import os

from _mrf import _ve_step, _concensus

TINY = 1e-300
NITERS = 20
ALPHA = 1.0
BETA = 0.2 

# VM-step 
def gauss_dist(x, mu, sigma):
    return np.exp(-.5*((x-float(mu))/float(sigma))**2)/float(sigma)

def laplace_dist(x, mu, sigma): 
    return np.exp(-np.abs((x-float(mu))/float(sigma)))/float(sigma)

def vm_step_gauss(ppm, data_, mask): 
    """
    ppm: ndarray (4d)
    data_: ndarray (1d, masked data)
    mask: 3-element tuple of 1d ndarrays (X,Y,Z)
    """
    nclasses = ppm.shape[3]
    mu = np.zeros(nclasses)
    sigma = np.zeros(nclasses)

    for i in range(nclasses):
        P = ppm[:,:,:,i][mask]
        Z = P.sum()
        tmp = data_*P
        mu_ = tmp.sum()/Z
        sigma_ = np.sqrt(np.sum(tmp*data_)/Z - mu_**2)
        mu[i] = mu_ 
        sigma[i] = sigma_
    return mu, sigma 


def weighted_median(x, w, ind): 
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
    nclasses = ppm.shape[3]
    mu = np.zeros(nclasses)
    sigma = np.zeros(nclasses)
    ind = np.argsort(data_) # data_[ind] increasing

    for i in range(nclasses):
        P = ppm[:,:,:,i][mask]
        mu_ = weighted_median(data_, P, ind) 
        sigma_ = np.sum(np.abs(P*(data_-mu_)))/P.sum()
        mu[i] = mu_ 
        sigma[i] = sigma_
    return mu, sigma 



class VEM(object): 
    """
    Classification via VEM algorithm.
    """

    def __init__(self, data, nclasses, mask=None, noise='gauss', 
                 ppm=None, copy=False, hard=False, 
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

        # No mask situation
        if mask == None: 
            mask = [slice(0, s) for s in data.shape]

        # If a ppm is provided, interpret it as a prior, otherwise
        # create ppm from scratch and assume flat prior.
        if ppm == None:
            self.ppm = np.zeros(list(data.shape)+[nclasses])
            self.ppm[mask] = 1/float(nclasses)
        else:
            self.ppm = ppm 
        self.mask = mask 
        self.prior_ = self.ppm[mask]
        self.data_ = data[mask]
        self.ref_ = np.zeros([self.data_.size, nclasses])
        self.nclasses = nclasses

        # Label information 
        if labels == None: 
            labels = [str(l) for l in range(nclasses)]
        if not len(labels) == self.nclasses: 
            raise ValueError('Wrong length for labels sequence') 
        self.labels = labels

        # Mixing matrix 
        self.mixmat = mixmat

        # Cached beta parameter
        self._beta = None

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
    def ve_step(self, mu, sigma, alpha=ALPHA, beta=BETA): 
        """
        VE-step
        """
        # Cache beta parameter
        self._beta = beta 

        # Compute complete-data likelihood maps, replacing very small
        # values for numerical stability
        for i in range(self.nclasses): 
            self.ref_[:,i] = self.prior_[:,i]**alpha
            self.ref_[:,i] *= self.dist(self.data_, mu[i], sigma[i])
        self.ref_[:] = np.maximum(self.ref_, TINY) 

        # Normalize reference probability map 
        if beta == 0.0: 
            self.ppm[self.mask] = (self.ref_.T/self.ref_.sum(1)).T

        # Update and normalize reference probabibility map using
        # neighborhood information (mean-field theory)
        else: 
            print('  ... MRF correction')
            # Deal with mixing matrix and label switching
            mixmat = self.mixmat 
            if not mixmat == None:
                mixmat = self.sort_mixmat(mu)
            self.ppm = _ve_step(self.ppm, self.ref_, 
                                np.array(self.mask, dtype='int'), 
                                beta, self.copy, self.hard, mixmat)
            

    def run(self, mu=None, sigma=None, alpha=ALPHA, beta=BETA, niters=NITERS): 

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
            self.ve_step(mu, sigma, alpha=alpha, beta=beta)
            do_vm_step = True

        return mu, sigma 


    def free_energy(self):
        """
        Compute the free energy defined as:

        F(q, theta) = int q(x) log q(x)/p(x,y/theta) dx

        associated with input parameters mu,
        sigma and beta (up to an ignored constant).
        """
        q_ = self.ppm[self.mask]
        # Entropy term
        f = np.sum(q_*np.log(np.maximum(q_/self.ref_, TINY)))
        # Interaction term
        if self._beta > 0.0: 
            print('  ... Concensus correction')
            fc = _concensus(self.ppm, np.array(self.mask, dtype='int'))
            f -= .5*self._beta*fc 
        return f
