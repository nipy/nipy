#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Script example of tissue classification
"""
from __future__ import print_function # Python 2/3 compatibility

import numpy as np

from nipy import load_image, save_image
from nipy.core.image.image_spaces import (make_xyz_image,
                                          xyz_affine)
from nipy.externals.argparse import ArgumentParser
from nipy.algorithms.segmentation import BrainT1Segmentation


def fuzzy_dice(gold_ppm, ppm, mask):
    """
    Fuzzy dice index.
    """
    dices = np.zeros(3)
    if gold_ppm == None:
        return dices
    for k in range(3):
        pk = gold_ppm[mask][:, k]
        qk = ppm[mask][:, k]
        PQ = np.sum(np.sqrt(np.maximum(pk * qk, 0)))
        P = np.sum(pk)
        Q = np.sum(qk)
        dices[k] = 2 * PQ / float(P + Q)
    return dices


# Parse command line
description = 'Perform brain tissue classification from skull stripped T1 \
image in CSF, GM and WM. If no mask image is provided, the mask is defined by \
thresholding the input image above zero (strictly).'

parser = ArgumentParser(description=description)
parser.add_argument('img', metavar='img', nargs='+', help='input image')
parser.add_argument('--mask', dest='mask', help='mask image')
parser.add_argument('--niters', dest='niters',
    help='number of iterations (default=%d)' % 25)
parser.add_argument('--beta', dest='beta',
    help='Markov random field beta parameter (default=%f)' % 0.5)
parser.add_argument('--ngb_size', dest='ngb_size',
    help='Markov random field neighborhood system (default=%d)' % 6)
parser.add_argument('--probc', dest='probc', help='csf probability map')
parser.add_argument('--probg', dest='probg',
    help='gray matter probability map')
parser.add_argument('--probw', dest='probw',
    help='white matter probability map')
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
mask_img = get_argument('mask', None)
if mask_img == None:
    mask_img = img
else:
    mask_img = load_image(mask_img)

# Other optional arguments
niters = int(get_argument('niters', 25))
beta = float(get_argument('beta', 0.5))
ngb_size = int(get_argument('ngb_size', 6))

# Perform tissue classification
mask = mask_img.get_data() > 0
S = BrainT1Segmentation(img.get_data(), mask=mask, model='5k',
                        niters=niters, beta=beta, ngb_size=ngb_size)

# Save label image
outfile = 'hard_classif.nii'
save_image(make_xyz_image(S.label, xyz_affine(img), 'scanner'),
           outfile)
print('Label image saved in: %s' % outfile)

# Compute fuzzy Dice indices if a 3-class fuzzy model is provided
if not args.probc == None and \
        not args.probg == None and \
        not args.probw == None:
    print('Computing Dice index')
    gold_ppm = np.zeros(S.ppm.shape)
    gold_ppm_img = (args.probc, args.probg, args.probw)
    for k in range(3):
        img = load_image(gold_ppm_img[k])
        gold_ppm[..., k] = img.get_data()
    d = fuzzy_dice(gold_ppm, S.ppm, np.where(mask_img.get_data() > 0))
    print('Fuzzy Dice indices: %s' % d)
