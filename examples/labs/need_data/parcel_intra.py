#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
Example of script to parcellate the data from one subject, using various
algorithms.

Note that it can take some time.

author: Bertrand Thirion, 2005-2009
"""
print(__doc__)

from os import mkdir, getcwd, path

from numpy import array

from nipy.labs.spatial_models.parcel_io import fixed_parcellation

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset

# ------------------------------------
# Get the data (mask+functional image)
# take several experimental conditions
# time courses could be used instead

n_beta = [29]
mask_image = path.join(DATA_DIR, 'mask.nii.gz')
betas = [path.join(DATA_DIR, 'spmT_%04d.nii.gz' % n) for n in n_beta]
missing_file = array([not path.exists(m) for m in [mask_image] + betas]).any()
if missing_file:
    get_second_level_dataset()

# set the parameters
n_parcels = 500
mu = 10
nn = 6
verbose = 1
# write directory
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)


lpa = fixed_parcellation(mask_image, betas, n_parcels, nn, 'gkm',
                         write_dir, mu, verbose)
lpa = fixed_parcellation(mask_image, betas, n_parcels, nn, 'ward',
                         write_dir, mu, verbose)
lpa = fixed_parcellation(mask_image, betas, n_parcels, nn, 'ward_and_gkm',
                         write_dir, mu, verbose)
