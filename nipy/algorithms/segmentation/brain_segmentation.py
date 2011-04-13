import numpy as np 

from nipy.core.image.affine_image import AffineImage

from .vem import VEM

NITERS = 25
BETA = 0.2
SCHEME = 'mf'
NOISE = 'gauss'
FREEZE_PROP = True
SYNCHRONOUS = False 
LABELS = ('CSF', 'GM', 'WM') 
VERBOSE = True


def initialize_parameters(data, klasses): 
    """
    Rough parameter initialization by moment matching with a brainweb
    image for which accurate parameters are known.
    """
    # Brainweb reference mean and standard devs
    ref_mu = np.array([813.9, 1628.4, 2155.8])
    ref_sigma = np.array([215.6, 173.9, 130.9])
    ##ref_prop = [.20, .47, .33]
    ref_glob_mean = 1643.1
    ref_glob_std = 502.8
    
    # Moment matching 
    x = np.linspace(0, 2, num=klasses) 
    prop = np.ones(len(x))/len(x)
    mu = np.zeros(len(x))
    sigma = np.zeros(len(x))

    I = x <= 1
    J = True - I 
    mu[I] = ref_mu[0] + x[I]*(ref_mu[1]-ref_mu[0])
    sigma[I] = ref_sigma[0] + x[I]*(ref_sigma[1]-ref_sigma[0])
    mu[J] = ref_mu[1] + (x[J]-1)*(ref_mu[2]-ref_mu[1])
    sigma[J] = ref_sigma[1] + (x[J]-1)*(ref_sigma[2]-ref_sigma[1])

    a = np.std(data) / ref_glob_std
    b = np.mean(data) - a*ref_glob_mean

    return a*mu + b, a*sigma, prop



def brain_segmentation(img, mask_img=None, hard=False, niters=NITERS, 
                       labels=LABELS, mixmat=None,  
                       noise=NOISE, beta=BETA, freeze_prop=FREEZE_PROP, 
                       scheme=SCHEME, synchronous=SYNCHRONOUS):
    
    """
    Perform tissue classification of a brain MR T1-weighted image into
    gray matter, white matter and CSF. The image needs be
    skull-stripped beforehand for the method to work. 

    Parameters
    ----------

    img : nipy-like image
      MR-T1 image to segment. 

    mask_img : nipy-like image 
      Brain mask. If None, the mask will be defined by thresholding
      the input image above zero (strictly).


    Returns
    -------

    ppm_img: nipy-like image 
      A 4D image representing the posterior probability map of each
      tissue.

    label_img: nipy-like image 
      Hard tissue classification image similar to a MAP. 

    """
    # Get an array out of the mask image 
    if mask_img == None: 
        mask_img = img
    mask = np.where(mask_img.get_data()>0)

    # Perform tissue classification
    mu, sigma, prop = initialize_parameters(img.get_data()[mask], len(labels))
    vem = VEM(img.get_data(), labels, mask=mask, scheme=scheme, noise=noise)
    mu, sigma, prop = vem.run(mu=mu, sigma=sigma, prop=prop, freeze_prop=freeze_prop, 
                              beta=beta, niters=niters)

    # Display information 
    if VERBOSE: 
        print('Estimated tissue means: %s' % mu) 
        print('Estimated tissue std deviates: %s' % sigma) 
        if not freeze_prop: 
            print('Estimated tissue proportions: %s' % prop) 

    # Sort and merge equivalent classes mixmat should be a matrix with
    # shape (K, 3), each row describing the probability of tissues
    # given the corresponding label.
    if mixmat == None: 
        ppm = vem.ppm
    else:
        ppm = np.zeros(list(img.shape)+[mixmat.shape[1]]) 
        ppm[mask] = np.dot(vem.ppm[mask], mixmat) 
    del vem 

    # Create output images 
    ppm_img = AffineImage(ppm, img.affine, 'scanner')
    pmode = np.zeros(img.shape, dtype='uint8')
    pmode[mask] = ppm[mask].argmax(1) + 1
    label_img = AffineImage(pmode, img.affine, 'scanner')

    return ppm_img, label_img
