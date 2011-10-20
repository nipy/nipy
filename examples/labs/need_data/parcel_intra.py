# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of script to parcellate the data from one subject,
using various algorithms.

Note that it can take some time.

author: Bertrand Thirion, 2005-2009
"""
print __doc__

import os.path as op
import tempfile
from numpy import array
from nipy.labs.spatial_models.parcel_io import fixed_parcellation

# ------------------------------------
# Get the data (mask+functional image)
# take several experimental conditions
# time courses could be used instead

n_beta = [29]
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data'))
mask_image = op.join(data_dir, 'mask.nii.gz')
betas = [op.join(data_dir, 'spmT_%04d.nii.gz' % n) for n in n_beta]
missing_file = array([not op.exists(m) for m in [mask_image] + betas]).any()
if missing_file:
    import get_data_light
    get_data_light.get_second_level_dataset()

# set the parameters
n_parcels = 500
mu = 10
nn = 6
write_dir = tempfile.mkdtemp()
verbose = 1

lpa = fixed_parcellation(mask_image, betas, n_parcels, nn, 'gkm',
                            write_dir, mu, verbose)
lpa = fixed_parcellation(mask_image, betas, n_parcels, nn, 'ward',
                            write_dir, mu, verbose)
lpa = fixed_parcellation(mask_image, betas, n_parcels, nn, 'ward_and_gkm',
                            write_dir, mu, verbose)
