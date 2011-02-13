# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Simple module to perform various tests at the voxel-level on images
"""

import numpy as np
from nipy.io.imageformats import load, save, Nifti1Image
from ..mask import intersect_masks


def ffx(maskImages, effectImages, varianceImages, resultImage=None):
    """ Computation of the fixed effects statistics on images

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
    # fixme : check that the images have same referntial
    # fixme : check that mask_Images is a list
    if len(effectImages) != len(varianceImages):
        raise ValueError('Not the correct number of images')
    tiny = 1.e-15
    nsubj = len(effectImages)
    mask = intersect_masks(maskImages, None, threshold=0.5, cc=True)

    effects = []
    variance = []
    for s in range(nsubj):
        rbeta = load(effectImages[s])
        beta = rbeta.get_data()[mask > 0]
        rbeta = load(varianceImages[s])
        varbeta = rbeta.get_data()[mask > 0]
        effects.append(beta)
        variance.append(varbeta)

    # clean the calues to avoid singularities
    effects = np.array(effects)
    variance = np.array(variance)
    effects[np.isnan(effects)] = 0
    effects[np.isnan(variance)] = 0
    variance[np.isnan(variance)] = tiny
    variance[variance == 0] = tiny

    # compute the t stat
    t = effects / np.sqrt(variance)
    t = t.mean(0) * np.sqrt(nsubj)

    nim = load(effectImages[0])
    affine = nim.get_affine()
    tmap = np.zeros(nim.get_shape())
    tmap[mask > 0] = t
    tImage = Nifti1Image(tmap, affine)
    if resultImage != None:
        save(tImage, resultImage)

    return tmap


def ffx_from_stat(maskImages, statImages, resultImage=None):
    """ Computation of the fixed effects statistics from statistics images

    Parameters
    ----------
    maskImages, string or list of strings
                the paths of one or several masks
                when several masks, the half thresholding heuristic is used
    statImages, list of strings
                the paths ofthe statitsic images
    resultImage: string, optional
                 path of the result images

    Returns
    -------
    the computed values
    """
    # fixme : check that the images have same referntial
    # fixme : check that mask_Images is a list
    nsubj = len(statImages)
    mask = intersect_masks(maskImages, None, threshold=0.5, cc=True)

    t = []
    for s in range(nsubj):
        rbeta = load(statImages[s])
        beta = rbeta.get_data()[mask > 0]
        t.append(beta)

    t = np.array(t)
    t[np.isnan(t)] = 0
    t = t.mean(0) * np.sqrt(nsubj)

    nim = load(statImages[0])
    affine = nim.get_affine()
    tmap = np.zeros(nim.get_shape())
    tmap[mask > 0] = t
    tImage = Nifti1Image(tmap, affine)
    if resultImage != None:
        save(tImage, resultImage)

    return tmap
