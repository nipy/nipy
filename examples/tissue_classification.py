#!/usr/bin/env python 
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from os.path import join, split
from argparse import ArgumentParser
import sys
from tempfile import mkdtemp
import numpy as np 

from nipy import load_image, save_image 
from nipy.core.image.affine_image import AffineImage 
from nipy.algorithms.segmentation import brain_segmentation

K_CSF = 1
K_GM = 1
K_WM = 1
NITERS = 10
BETA = 0.2
SCHEME = 'mf'
NOISE = 'gauss'
FREEZE_PROP = True


def fuzzy_dice(gpm, ppm, mask):
    """
    Fuzzy dice index. 
    """
    dices = np.zeros(3)
    if gpm == None: 
        return dices
    for k in range(3): 
        pk = gpm[k].get_data()[mask]
        qk = ppm.get_data()[mask][:,k]
        PQ = np.sum(np.sqrt(np.maximum(pk*qk, 0)))
        P = np.sum(pk)
        Q = np.sum(qk)
        dices[k] = 2*PQ/float(P+Q)
    return dices



# Parse command line 
description = 'Perform brain tissue classification from skull stripped T1 image in \
CSF, GM and WM. If no mask image is provided, the mask is defined by \
thresholding the input image above zero (strictly).'

parser = ArgumentParser(description=description)
parser.add_argument('img', metavar='img', nargs='+', help='input image')
parser.add_argument('--mask', dest='mask', help='mask image')
parser.add_argument('--k_csf', dest='k_csf', 
                    help='number of CSF classes (default=%d)' % K_CSF)
parser.add_argument('--k_gm', dest='k_gm', 
                    help='number of GM classes (default=%d)' % K_GM)
parser.add_argument('--k_wm', dest='k_wm', 
                    help='number of WM classes (default=%d)' % K_WM)
parser.add_argument('--niters', dest='niters', 
                    help='number of iterations (default=%d)' % NITERS)
parser.add_argument('--beta', dest='beta', 
                    help='Markov random field beta parameter (default=%f)' % BETA)
parser.add_argument('--scheme', dest='scheme', 
                    help='message passing scheme (mf, icm or bp, default=%s)' % SCHEME)
parser.add_argument('--noise', dest='noise', 
                    help='noise model (gauss or laplace, default=%s)' % NOISE)
parser.add_argument('--freeze_prop', dest='freeze_prop', 
                    help='freeze tissue proportions (default=1)')
parser.add_argument('--probc', dest='probc', help='csf probability map')
parser.add_argument('--probg', dest='probg', help='gray matter probability map')
parser.add_argument('--probw', dest='probw', help='white matter probability map')

args = parser.parse_args() 

def get_argument(dest, default): 
    val = args.__getattribute__(dest)
    if val == None: 
        return default
    else: 
        return val 

# Input image
img = load_image(args.img[0])

# Input mask image 
mask = get_argument('mask', None)
if mask == None: 
    mask_img = img
else:
    mask_img = load_image(mask)

# Other optional arguments
k_csf = int(get_argument('k_csf', K_CSF))
k_gm = int(get_argument('k_gm', K_GM))
k_wm = int(get_argument('k_wm', K_WM))
niters = int(get_argument('niters', NITERS))
beta = float(get_argument('beta', BETA))
scheme = get_argument('scheme', SCHEME)
noise = get_argument('noise', NOISE)
freeze_prop = get_argument('freeze_prop', FREEZE_PROP)

# Perform tissue classification
ppm_img, label_img = brain_segmentation(img, mask_img=mask_img, beta=beta, niters=niters, 
                                        k_csf=k_csf, k_gm=k_gm, k_wm=k_wm,
                                        noise=noise, freeze_prop=freeze_prop, 
                                        scheme=SCHEME)
outfile = join(mkdtemp(), 'hard_classif.nii')
save_image(label_img, outfile)
print('Label image saved in: %s' % outfile) 

# Compute fuzzy Dice indices if a 3-class fuzzy model is provided 
if not args.probc == None and not args.probg == None and not args.probw == None:
    print('Computing Dice index') 
    gpm = [load_image(args.probc), load_image(args.probg), load_image(args.probw)]
    d = fuzzy_dice(gpm, ppm_img, np.where(mask_img.get_data()>0))
    print('Fuzzy Dice indices: %s' % d) 


