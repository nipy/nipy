#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function
__doc__ = \
"""
This example shows how to get variance and beta estimated from a nipy GLM.

More specifically:

1. A sequence of fMRI volumes are loaded.
2. A design matrix describing all the effects related to the data is computed.
3. A GLM is applied to the dataset, effect and variance images are produced.

Note that this corresponds to a single run.

Needs matplotlib

Author : Bertrand Thirion, 2010--2012
"""
print(__doc__)

from os import mkdir, getcwd, path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nibabel import Nifti1Image, save

from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import \
    load_paradigm_from_csv_file
from nipy.labs.viz import plot_map, cm

# Local import
from get_data_light import DATA_DIR, get_first_level_dataset

#######################################
# Data and analysis parameters
#######################################

# volume mask
# This dataset is large
get_first_level_dataset()
data_path = path.join(DATA_DIR, 's12069_swaloc1_corr.nii.gz')
paradigm_file = path.join(DATA_DIR, 'localizer_paradigm.csv')

# timing
n_scans = 128
tr = 2.4

# paradigm
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)

# confounds
hrf_model = 'canonical'
drift_model = "cosine"
hfcut = 128

# write directory
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

print('Computation will be performed in directory: %s' % write_dir)

########################################
# Design matrix
########################################

print('Loading design matrix...')

# the example example.labs.write_paradigm_file shows how to create this file
paradigm = load_paradigm_from_csv_file(paradigm_file)['0']

design_matrix = make_dmtx(frametimes, paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')

plt.savefig(path.join(write_dir, 'design_matrix.png'))
dim = design_matrix.matrix.shape[1]

########################################
# Perform a GLM analysis
########################################

print('Fitting a GLM (this takes time)...')
fmri_glm = FMRILinearModel(data_path, design_matrix.matrix,
                           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

########################################
# Output beta and variance images
########################################
beta_hat = fmri_glm.glms[0].get_beta()  # Least-squares estimates of the beta
variance_hat = fmri_glm.glms[0].get_mse() # Estimates of the variance
mask = fmri_glm.mask.get_data() > 0

# output beta images
beta_map = np.tile(mask.astype(np.float)[..., np.newaxis], dim)
beta_map[mask] = beta_hat.T
beta_image = Nifti1Image(beta_map, fmri_glm.affine)
beta_image.get_header()['descrip'] = (
    'Parameter estimates of the localizer dataset')
save(beta_image, path.join(write_dir, 'beta.nii'))
print("Beta image witten in %s" % write_dir)

variance_map = mask.astype(np.float)
variance_map[mask] = variance_hat

# Create a snapshots of the variance image contrasts
vmax = np.log(variance_hat.max())
plot_map(np.log(variance_map + .1),
         fmri_glm.affine,
         cmap=cm.hot_black_bone,
         vmin=np.log(0.1),
         vmax=vmax,
         anat=None,
         threshold=.1, alpha=.9)
plt.show()
