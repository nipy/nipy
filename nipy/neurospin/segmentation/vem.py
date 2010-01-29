import numpy as np 
import pylab 
import os

from mrf_module import finalize_ve_step



# VM-step 
def vm_step_gauss(ppm, data, mask): 
    """
    ppm: ndarray (4d)
    data: ndarray (1d, masked data)
    mask: 3-element tuple of 1d ndarrays (X,Y,Z)
    """
    ntissues = ppm.shape[3]
    mu = np.zeros(ntissues)
    sigma = np.zeros(ntissues)

    for i in range(ntissues):
        P = ppm[:,:,:,i][mask]
        Z = P.sum()
        tmp = data*P
        mu_ = tmp.sum()/Z
        sigma_ = np.sqrt(np.sum(tmp*data)/Z - mu_**2)
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

def vm_step_laplace(ppm, data, mask): 
    """
    ppm: ndarray (4d)
    data: ndarray (1d, masked data)
    mask: 3-element tuple of 1d ndarrays (X,Y,Z)
    """
    ntissues = ppm.shape[3]
    mu = np.zeros(ntissues)
    sigma = np.zeros(ntissues)
    ind = np.argsort(data) # data[ind] increasing

    for i in range(ntissues):
        P = ppm[:,:,:,i][mask]
        mu_ = wmedian(data, P, ind) 
        sigma_ = np.sum(np.abs(P*(data-mu_)))/P.sum()
        mu[i] = mu_ 
        sigma[i] = sigma_
    return mu, sigma 




# VE-step 
def ve_step(ppm, data, mask, mu, sigma, prior, ndist, alpha=1., beta=0.0): 
    """
    posterior = e_step(gaussians, prior, data, posterior=None)    

    data are assumed masked. 
    """
    ntissues = ppm.shape[3]
    lik = np.zeros(np.shape(prior))
    for i in range(ntissues): 
        lik[:,i] = prior[:,i] * ndist(data, mu[i], sigma[i])

    # Normalize
    X, Y, Z = mask
    ppm[X, Y, Z] = lik 

    if beta == 0.0: 
        ppm_sum = ppm[mask].sum(1)
        for i in range(ntissues): 
            ppm[X, Y, Z, i] /= ppm_sum
 
    else: 
        print('  .. MRF correction')
        XYZ = np.array((X, Y, Z), dtype='int') 
        ppm = finalize_ve_step(ppm, lik, XYZ, beta)

    return ppm
        

# VEM algorithm 
def vem(ppm, data, mask, prior, alphas=None, betas=None, niters=5, 
        noise='gauss'): 
    """
    ppm: ndarray (4d)
    data: ndarray (1d, masked data)
    mask: 3-element tuple of 1d ndarrays (X,Y,Z)
    prior: ndarray (2d, masked data)
    """

    if betas == None: 
        betas = np.zeros(niters)
    else:
        niters = len(betas)
    if alphas == None: 
        alphas = np.ones(niters)
    else:
        if not len(alphas) == niters:
            raise ValueError('Inconsistent length for alphas and betas.')

    if noise == 'gauss': 
        vm_step = vm_step_gauss
        def ndist(x, mu, sigma): 
            return np.exp(-.5*((x-mu)/sigma)**2)/sigma
    elif noise == 'laplace':
        vm_step = vm_step_laplace
        def ndist(x, mu, sigma): 
            return np.exp(-np.abs((x-mu)/sigma))/sigma
    else:
        raise ValueError('Unknown noise model')

    for i in range(niters):
        print('VEM iter %d/%d' % (i+1, niters))
        print('  VM-step...')
        mu, sigma = vm_step(ppm, data, mask) 
        print('  VE-step...')
        ppm = ve_step(ppm, data, mask, mu, sigma, prior, 
                      ndist, alpha=alphas[i], beta=betas[i]) 
        
    return ppm, mu, sigma



