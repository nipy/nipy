#!/usr/bin/env python 
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from os.path import join, split
from optparse import OptionParser
import sys
from tempfile import mkdtemp
import numpy as np 

from nipy import load_image, save_image 
from nipy.core.image.affine_image import AffineImage 
from nipy.algorithms.segmentation import VEM

LABELS = ('CSF','GM','WM')
NITERS = 10
BETA = 0.2
SCHEME = 'mf'
PROP = [.33, .33, .33]
FREEZE_PROP = True


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

def fuzzy_dice(gpm, ppm, labels, mask):
    """
    Fuzzy dice index. 
    """
    dices = np.zeros(len(LABELS))
    if gpm == None: 
        return dices
    for kk in range(len(LABELS)): 
        tissue = labels[kk]
        k = LABELS.index(tissue)
        pk = gpm[k].get_data()[mask]
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


# Parse command line 
usage = 'usage: %prog [options] img_file'

description = 'Perform brain tissue classification from skull stripped T1 image in \
CSF, GM and WM. If no mask image is provided, the mask is defined by \
thresholding the input image above zero (strictly).'

parser = OptionParser(usage=usage, description=description)
parser.add_option('-n', '--niters', dest='niters', 
                  help='number of iterations (default=%d)' % NITERS)
parser.add_option('-b', '--beta', dest='beta', 
                  help='Markov random field beta parameter (default=%f)' % BETA)
parser.add_option('-s', '--scheme', dest='scheme', 
                  help='message passing scheme (mf, icm or bp, default=%s)' % SCHEME)

parser.add_option('-m', '--mask', dest='mask', help='mask image')
parser.add_option('-c', '--probc', dest='probc', help='csf probability map')
parser.add_option('-g', '--probg', dest='probg', help='gray matter probability map')
parser.add_option('-w', '--probw', dest='probw', help='white matter probability map')
opts, args = parser.parse_args() 

# Input image
if len(args)>0: 
    im = load_image(args[0])
else: 
    print('Missing input image.') 
    sys.exit()

# Number of iterations 
if opts.niters == None:
    niters = NITERS
else:
    niters = int(opts.niters)

# Beta parameter
if opts.beta == None: 
    beta = BETA
else:
    beta = float(opts.beta)

# Message passing scheme
if opts.scheme == None:
    scheme = SCHEME
else:
    scheme = opts.scheme

# Input mask image 
if opts.mask == None: 
    mask = np.where(im.get_data()>0)
else:
    mask_im = load_image(opts.mask)
    mask = np.where(mask_im.get_data()>0)

# Perform tissue classification
mu, sigma = moment_matching(im, mask)
vem = VEM(im.get_data(), 3, mask=mask, labels=LABELS, scheme=scheme)
mu, sigma, prop = vem.run(mu=mu, sigma=sigma, 
                          prop=PROP, freeze_prop=FREEZE_PROP, 
                          beta=beta, niters=niters)

print vem.free_energy() 

# Display information 
print('Estimated tissue means: %s' % mu) 
print('Estimated tissue std deviates: %s' % sigma) 
print('Estimated tissue proportions: %s' % prop) 

# Generate hard tissue classification image 
classif = hard_classification(im, mask, vem.ppm) 
outfile = join(mkdtemp(), 'hard_classif.nii')
save_image(classif, outfile)
print('Label image saved in: %s' % outfile) 

# Compute fuzzy Dice indices if a complete prob model is provided 
if not opts.probc == None and not opts.probg == None and not opts.probw == None:
    print('Computing Dice index') 
    gpm = [load_image(opts.probc), load_image(opts.probg), load_image(opts.probw)]
    d = fuzzy_dice(gpm, vem.ppm, LABELS, mask)
    print('Fuzzy Dice indices: %s' % d) 



