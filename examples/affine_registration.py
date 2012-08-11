#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This script requires the nipy-data package to run. It is an example of
inter-subject affine registration using two MR-T1 images from the
sulcal 2000 database acquired at CEA, SHFJ, Orsay, France. The source
is 'ammon' and the target is 'anubis'. Running it will result in a
resampled ammon image being created in the current directory.
"""

from optparse import OptionParser
import time

import numpy as np

from nipy.algorithms.registration import HistogramRegistration, resample
from nipy.utils import example_data
from nipy import load_image, save_image

print('Scanning data directory...')

# Input images are provided with the nipy-data package
source = 'ammon'
target = 'anubis'
source_file = example_data.get_filename('neurospin', 'sulcal2000',
                                        'nobias_' + source + '.nii.gz')
target_file = example_data.get_filename('neurospin', 'sulcal2000',
                                        'nobias_' + target + '.nii.gz')

# Parse arguments
parser = OptionParser(description=__doc__)

doc_similarity = 'similarity measure: cc (correlation coefficient), \
cr (correlation ratio), crl1 (correlation ratio in L1 norm), \
mi (mutual information), nmi (normalized mutual information), \
pmi (Parzen mutual information), dpmi (discrete Parzen mutual \
information). Default is crl1.'

doc_interp = 'interpolation method: tri (trilinear), pv (partial volume), \
rand (random). Default is pv.'

doc_optimizer = 'optimization method: simplex, powell, steepest, cg, bfgs. \
Default is powell.'

parser.add_option('-s', '--similarity', dest='similarity',
                  help=doc_similarity)
parser.add_option('-i', '--interp', dest='interp',
                  help=doc_interp)
parser.add_option('-o', '--optimizer', dest='optimizer',
                  help=doc_optimizer)
opts, args = parser.parse_args()


# Optional arguments
similarity = 'crl1'
interp = 'pv'
optimizer = 'powell'
if not opts.similarity == None:
    similarity = opts.similarity
if not opts.interp == None:
    interp = opts.interp
if not opts.optimizer == None:
    optimizer = opts.optimizer

# Print messages
print ('Source brain: %s' % source)
print ('Target brain: %s' % target)
print ('Similarity measure: %s' % similarity)
print ('Optimizer: %s' % optimizer)

# Get data
print('Fetching image data...')
I = load_image(source_file)
J = load_image(target_file)

# Perform affine registration
# The output is an array-like object such that
# np.asarray(T) is a customary 4x4 matrix
print('Setting up registration...')
tic = time.time()
R = HistogramRegistration(I, J, similarity=similarity, interp=interp)
T = R.optimize('affine', optimizer=optimizer)
toc = time.time()
print('  Registration time: %f sec' % (toc - tic))

# Resample source image
print('Resampling source image...')
tic = time.time()
#It = resample2(I, J.coordmap, T.inv(), J.shape)
It = resample(I, T.inv(), reference=J)
toc = time.time()
print('  Resampling time: %f sec' % (toc - tic))

# Save resampled source
outroot = source + '_TO_' + target
outimg = outroot + '.nii.gz'
print ('Saving resampled source in: %s' % outimg)
save_image(It, outimg)

# Save transformation matrix
outparams = outroot + '.npy'
np.save(outparams, np.asarray(T))
