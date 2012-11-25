#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
"""
Script that performs a first-level analysis of the FIAC dataset.

See ``examples/fiac/fiac_example.py`` for another approach to this analysis.

Needs the *example data* package.

Also needs matplotlib

Author: Alexis Roche, Bertrand Thirion, 2009--2012
"""

from os import mkdir, getcwd, path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nibabel import save

from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.utils import example_data
from nipy.labs.viz import plot_map, cm

# -----------------------------------------------------------
# --------- Get the data -----------------------------------
#-----------------------------------------------------------

fmri_files = [example_data.get_filename('fiac', 'fiac0', run)
              for run in ['run1.nii.gz', 'run2.nii.gz']]
design_files = [example_data.get_filename('fiac', 'fiac0', run)
                for run in ['run1_design.npz', 'run2_design.npz']]
mask_file = example_data.get_filename('fiac', 'fiac0', 'mask.nii.gz')

# Load all the data
multi_session_model = FMRILinearModel(fmri_files, design_files, mask_file)

# GLM fitting
multi_session_model.fit(do_scaling=True, model='ar1')


def make_fiac_contrasts(p):
    """Specify some contrasts for the FIAC experiment

    Parameters
    ==========
    p: int, the number of columns of the design matrix (for all sessions)
    """
    con = {}
    # the design matrices of both runs comprise 13 columns
    # the first 5 columns of the design matrices correspond to the following
    # conditions: ["SSt-SSp", "SSt-DSp", "DSt-SSp", "DSt-DSp", "FirstSt"]

    def length_p_vector(con, p):
        return np.hstack((con, np.zeros(p - len(con))))

    con["SStSSp_minus_DStDSp"] = length_p_vector([1, 0, 0, - 1], p)
    con["DStDSp_minus_SStSSp"] = length_p_vector([- 1, 0, 0, 1], p)
    con["DSt_minus_SSt"] = length_p_vector([- 1, - 1, 1, 1], p)
    con["DSp_minus_SSp"] = length_p_vector([- 1, 1, - 1, 1], p)
    con["DSt_minus_SSt_for_DSp"] = length_p_vector([0, - 1, 0, 1], p)
    con["DSp_minus_SSp_for_DSt"] = length_p_vector([0, 0, - 1, 1], p)
    con["Deactivation"] = length_p_vector([- 1, - 1, - 1, - 1, 4], p)
    con["Effects_of_interest"] = np.eye(p)[:5]
    return con


# compute fixed effects of the two runs and compute related images
n_regressors = np.load(design_files[0])['X'].shape[1]
# note: implictly assume the same shape for all sessions !
contrasts = make_fiac_contrasts(n_regressors)

# write directory
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

print('Computing contrasts...')
mean_map = multi_session_model.means[0]  # for display
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' % (
        index + 1, len(contrasts), contrast_id))
    z_image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
    z_map, = multi_session_model.contrast(
        [contrast_val] * 2, con_id=contrast_id, output_z=True)
    save(z_map, z_image_path)

    # make a snapshot of the contrast activation
    if contrast_id == 'Effects_of_interest':
        vmax = max(- z_map.get_data().min(), z_map.get_data().max())
        vmin = - vmax
        plot_map(z_map.get_data(), z_map.get_affine(),
                 anat=mean_map.get_data(), anat_affine=mean_map.get_affine(),
                 cmap=cm.cold_hot,
                 vmin=vmin,
                 vmax=vmax,
                 figure=10,
                 threshold=2.5,
                 black_bg=True)
        plt.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

print("All the  results were witten in %s" % write_dir)
plt.show()
