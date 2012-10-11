#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function
__doc__ = """
Example of a one-sample t-test using the GLM formalism.
This script takes individual contrast images and masks and runs a simple GLM.
This can be readily generalized to any design matrix.

This particular example shows the statical map of a contrast
related to a computation task
(subtraction of computation task minus sentence reading/listening).

Needs matplotlib.

Author : Bertrand Thirion, 2012
"""
print(__doc__)

#autoindent
from os import mkdir, getcwd, path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nibabel import load, concat_images, save, Nifti1Image

from nipy.labs.mask import intersect_masks
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs.viz import plot_map, cm

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset

# Get the data
n_subjects = 12
n_beta = 29
data_dir = path.join(DATA_DIR, 'group_t_images')
mask_images = [path.join(data_dir, 'mask_subj%02d.nii' % n)
               for n in range(n_subjects)]

betas = [path.join(data_dir, 'spmT_%04d_subj_%02d.nii' % (n_beta, n))
         for n in range(n_subjects)]

missing_files = np.array([not path.exists(m) for m in mask_images + betas])
if missing_files.any():
    get_second_level_dataset()

write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

# Compute a population-level mask as the intersection of individual masks
grp_mask = Nifti1Image(intersect_masks(mask_images).astype(np.int8),
                       load(mask_images[0]).get_affine())

# concatenate the individual images
first_level_image = concat_images(betas)

# set the model
design_matrix = np.ones(len(betas))[:, np.newaxis]  # only the intercept
grp_model = FMRILinearModel(first_level_image, design_matrix, grp_mask)

# GLM fitting using ordinary least_squares
grp_model.fit(do_scaling=False, model='ols')

# specify and estimate the contrast
contrast_val = np.array(([[1]]))  # the only possible contrast !
z_map, = grp_model.contrast(contrast_val, con_id='one_sample', output_z=True)

# write the results
save(z_map, path.join(write_dir, 'one_sample_z_map.nii'))

# look at the result
vmax = max(- z_map.get_data().min(), z_map.get_data().max())
vmin = - vmax
plot_map(z_map.get_data(), z_map.get_affine(),
         cmap=cm.cold_hot,
         vmin=vmin,
         vmax=vmax,
         threshold=3.,
         black_bg=True)
plt.savefig(path.join(write_dir, '%s_z_map.png' % 'one_sample'))
plt.show()
print("Wrote all the results in directory %s" % write_dir)
