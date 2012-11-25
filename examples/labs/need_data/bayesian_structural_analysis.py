#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
Example of a script that uses the BSA (Bayesian Structural Analysis) i.e.
nipy.labs.spatial_models.bayesian_structural_analysis module.

Author : Bertrand Thirion, 2008-2010
"""
print(__doc__)

#autoindent
from os import mkdir, getcwd, path
import pickle

from numpy import array
from scipy import stats

from nipy.labs.spatial_models.bsa_io import make_bsa_image

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset

# Get the data
nbsubj = 12
nbeta = 29
data_dir = path.join(DATA_DIR, 'group_t_images')
mask_images = [path.join(data_dir, 'mask_subj%02d.nii' % n)
               for n in range(nbsubj)]

betas = [path.join(data_dir, 'spmT_%04d_subj_%02d.nii' % (nbeta, n))
         for n in range(nbsubj)]

missing_file = array([not path.exists(m) for m in mask_images + betas]).any()

if missing_file:
    get_second_level_dataset()

# set various parameters
subj_id = ['%04d' % i for i in range(12)]
theta = float(stats.t.isf(0.01, 100))
dmax = 4.
ths = 0
thq = 0.95
verbose = 1
smin = 5
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)
method = 'quick'
print('method used:', method)

# call the function
AF, BF = make_bsa_image(mask_images, betas, theta, dmax, ths, thq, smin,
                        write_dir, method, subj_id, '%04d' % nbeta,
                        reshuffle=False)

# Write the result. OK, this is only a temporary solution
picname = path.join(write_dir, "AF_%04d.pic" % nbeta)
pickle.dump(AF, open(picname, 'wb'), 2)
picname = path.join(write_dir, "BF_%04d.pic" % nbeta)
pickle.dump(BF, open(picname, 'wb'), 2)

print("Wrote all the results in directory %s" % write_dir)
