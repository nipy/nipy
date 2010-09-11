# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of script to parcellate the data from one subject,
using various algorithms.

Note that it can take some time.

author: Bertrand Thirion, 2005-2009
"""
print __doc__

import os.path
import tempfile

from nipy.neurospin.spatial_models.parcel_io import one_subj_parcellation

import get_data_light
data_dir = get_data_light.get_it()

# ------------------------------------
# Get the data (mask+functional image)
# take several experimental conditions
# time courses could be used instead

n_beta = [29]
mask_image = os.path.join(data_dir, 'mask.nii.gz')
betas = [os.path.join(data_dir, 'spmT_%04d.nii.gz' % n) for n in n_beta]

# set the parameters
n_parcels = 500
mu = 10
nn = 6
write_dir = tempfile.mkdtemp()
verbose = 1

lpa = one_subj_parcellation(mask_image, betas, n_parcels, nn, 'gkm', 
                            write_dir, mu, verbose)
lpa = one_subj_parcellation(mask_image, betas, n_parcels, nn, 'ward', 
                            write_dir, mu, verbose)
lpa = one_subj_parcellation(mask_image, betas, n_parcels, nn, 'ward_and_gkm', 
                            write_dir, mu, verbose)
