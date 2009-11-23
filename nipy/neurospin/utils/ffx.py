"""
Simple module to perform various tests at the voxel-level on images
"""

import numpy as np
from nipy.io.imageformats import load, save, Nifti1Image
from nipy.neurospin.utils.mask import intersect_masks

def ffx( maskImages, effectImages, varianceImages, resultImage=None):
    """
    Computation of the fixed effecst statistics

    Parameters
    ----------
    maskImages, string or list of strings
                the paths of one or several masks
                when several masks, the half thresholding heuristic is used
    effectImages, list of strings
                the paths ofthe effect images   
    varianceImages, list of strings
                    the paths of the associated variance images
    resultImage=None, string,
                 path of the result images

    Returns
    -------
    the computed values
    """
    # fixme : check that the images have same referntail
    # fixme : check that mask_Images is a list
    if len(effectImages)!=len(varianceImages):
        raise ValueError, 'Not the correct number of images'
    tiny = 1.e-15
    nsubj = len(effectImages)
    mask = intersect_masks(maskImages, None, threshold=0.5, cc=True)
    
    effects = []
    variance = []
    for s in range(nsubj):
        rbeta = load(effectImages[s])
        beta = rbeta.get_data()[mask>0]
        rbeta = load(varianceImages[s])
        varbeta = rbeta.get_data()[mask>0]
        
        effects.append(beta)
        variance.append(varbeta)
    
    effects = np.array(effects)
    variance = np.array(variance)
    effects[np.isnan(effects)] = 0
    effects[np.isnan(variance)] = 0
    variance[np.isnan(variance)] = tiny
    variance[variance==0] = tiny    
    
    t = effects/np.sqrt(variance)
    t = t.mean(0)*np.sqrt(nsubj)     
    #t = np.sum(effects/variance,0)/np.sum(1.0/np.sqrt(variance),0)

    nim = load(effectImages[0])
    affine = nim.get_affine()
    tmap = np.zeros(nim.get_shape())
    tmap[mask>0] = t
    tImage = Nifti1Image(tmap, affine)
    if resultImage!=None:
       save(tImage,resultImage)

    return tmap
