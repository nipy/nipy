import numpy as np 
import pylab 
import os

from nipy.neurospin.image import Image, load_image, save_image
from nipy.neurospin.segmentation.mrf_module import finalize_ve_step


# Display function 
def display(array, slice=80, threshold=None): 
    pylab.figure()
    if threshold==None: 
        pylab.imshow(array[:,slice,:])
    else:
        pylab.imshow(array[:,slice,:]>threshold)

# Load ppm
def load_ppm(): 
    # Read input files 
    for i in range(ntissues):
        fname = os.path.join(datadir, tissues[i]+'_1000Prior.img')
        if i == 0: 
            im = load_image(fname)
            affine = im.affine
            data = np.zeros(list(im.shape)+[ntissues])
            data[:,:,:,0] = im.data
        else: 
            data[:,:,:,i] = load_image(fname).data
    
    #ppm = Image(data, affine)

    # Normalize and mask ppms 
    psum = data.sum(3)
    X,Y,Z = np.where(psum>0)
    for i in range(ntissues): 
        data[X,Y,Z,i] /= psum[X,Y,Z]
    mask = (X.astype('uint'), Y.astype('uint'), Z.astype('uint'))

    return data, mask, affine 

# Load (bias corrected) mri scan 
def load_mri(): 
    fname = os.path.join(datadir, 'BiasCorIm.img')
    return load_image(fname)

# VM-step 
def vm_step(ppm, data, mask): 
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

def gaussian(x, mu, sigma): 
    return np.exp(-.5*((x-mu)/sigma)**2)/sigma

# VE-step 
def ve_step(ppm, data, mask, mu, sigma, prior, alpha=1., beta=0.0): 
    """
    posterior = e_step(gaussians, prior, data, posterior=None)    

    data are assumed masked. 
    """
    lik = np.zeros(np.shape(prior))
    for i in range(ntissues): 
        lik[:,i] = prior[:,i] * gaussian(data, mu[i], sigma[i])

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
def vem(ppm, data, mask, prior, alphas=None, betas=None, niters=5): 

    if not betas: 
        betas = np.zeros(niters)
    if not alphas: 
        alphas = np.zeros(niters)
    niters = len(betas)
    if not len(alphas) == niters:
        raise ValueError('Inconsistent length for alphas and betas.')

    for i in range(niters):
        print('VEM iter %d/%d' % (i+1, niters))
        print('  VM-step...')
        mu, sigma = vm_step(ppm, data, mask) 
        print('  VE-step...')
        ppm = ve_step(ppm, data, mask, mu, sigma, prior, 
                      alpha=alphas[i], beta=betas[i]) 
        


"""
ppm, mask, _ = load_ppm()
im = load_mri()
data = im.data[mask] 
affine = im.affine
prior = ppm[mask]
"""


