#!/usr/bin/env python 
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Perform brain tissue classification from skull stripped T1 image in
CSF, GM and WM.

Usage: 
  python tissue_classification img_file [niters][mask_file]

If no mask image is provided, the mask is defined by thresholding the
input image above zero (strictly). 
"""

from os.path import join, split
import sys
import tempfile 
import numpy as np 

from nipy import load_image, save_image 
from nipy.core.image.affine_image import AffineImage 
from nipy.algorithms.segmentation import VEM


LABELS = ('CSF','GM','WM')

def moment_matching(im, mask):
    """
    Rough parameter initialization by moment matching with a brainweb
    image for which accurate parameters are known.
    """
    mu_ = np.array([813.9, 1628.4, 2155.8])
    sigma_ = np.array([215.6, 173.9, 130.9])
    m_ = 1643.1
    s_ = 502.8
    data = im.get_data()[mask]
    m = np.mean(data)
    s = np.std(data)
    a = s/s_
    b = m - a*m_
    return a*mu_ + b, a*sigma_ 

def fuzzy_dice(gt, ppm, labels, mask):
    """
    Fuzzy dice index. 
    """
    dices = np.zeros(len(LABELS))
    if gt == None: 
        return dices
    for kk in range(len(LABELS)): 
        tissue = labels[kk]
        k = LABELS.index(tissue)
        pk = gt[k].get_data()[mask]
        qk = ppm[mask][:,kk]
        PQ = np.sum(np.sqrt(np.maximum(pk*qk, 0)))
        P = np.sum(pk)
        Q = np.sum(qk)
        dices[k] = 2*PQ/float(P+Q)
    return dices

def hard_classification(im, mask, ppm): 
    tmp = np.zeros(im.shape, dtype='uint8')
    tmp[mask] = ppm[mask].argmax(1) + 1
    return AffineImage(tmp, im.affine, 'scanner')


if len(sys.argv)<2:
    raise ValueError('Missing input image.') 
im = load_image(sys.argv[1])
if len(sys.argv)>2: 
    niters = int(sys.argv[2])
    magic = niters<0
    niters = abs(niters) 
if len(sys.argv)>3:
    mask_im = load_image(sys.argv[3])
    mask = np.where(mask_im.get_data()>0)
else:
    mask = np.where(im.get_data()>0)

mu, sigma = moment_matching(im, mask)
vem = VEM(im.get_data(), 3, mask=mask, labels=LABELS)
vem.run(mu=mu, sigma=sigma, prop=[.20, .47, .33], freeze_prop=False, niters=niters, beta=1.2)
classif = hard_classification(im, mask, vem.ppm) 

outfile = join(tempfile.mkdtemp(), 'classif.nii')
save_image(classif, outfile)
print('Label image saved in: %s' % outfile) 

############################################################
if magic: 
    print('Computing Dice index') 
    datadir, _ = split(sys.argv[1]) 
    prefix = 'phantom_1.0mm_normal_'
    fuzzy_model = [prefix+t+'_.nii' for t in ('csf','gry','wht')]
    gt = [load_image(join(datadir, f)) for f in fuzzy_model]
    d = fuzzy_dice(gt, vem.ppm, LABELS, mask)
    print('Fuzzy Dice indices: %s' % d) 



