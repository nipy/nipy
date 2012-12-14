#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function
__doc__ = """
Example where the result of the min of two contrasts is computed and displayed.
This is based on the Localizer dataset, in which we want to find the regions
activated both in left and right finger tapping.

Notes
-----
This is the valid conjunction test discussed in:
Nichols T, Brett M, Andersson J, Wager T, Poline JB. Valid conjunction
inference with the minimum statistic. Neuroimage. 2005 Apr 15;25(3):653-60.

Needs matplotlib

Author : Bertrand Thirion, 2012
"""
print(__doc__)

from os import mkdir, getcwd, path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nibabel import save

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
drift_model = 'cosine'
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

paradigm = load_paradigm_from_csv_file(paradigm_file)['0']

design_matrix = make_dmtx(frametimes, paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

#########################################
# Specify the contrasts
#########################################

# simplest ones
contrasts = {}
n_columns = len(design_matrix.names)
for i in range(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[i]] = np.eye(n_columns)[i]

# and more complex/ interesting ones
contrasts['left'] = contrasts['clicGaudio'] + contrasts['clicGvideo']
contrasts['right'] = contrasts['clicDaudio'] + contrasts['clicDvideo']

########################################
# Perform a GLM analysis
########################################

print('Fitting a General Linear Model')
fmri_glm = FMRILinearModel(data_path, design_matrix.matrix,
                           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

#########################################
# Estimate the contrasts
#########################################

contrast_id = 'left_right_motor_min'
z_map, effects_map = fmri_glm.contrast(
    np.vstack((contrasts['left'], contrasts['right'])), 
    contrast_type='tmin-conjunction', output_z=True, output_effects=True)
z_image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
save(z_map, z_image_path)

contrast_path = path.join(write_dir, '%s_con.nii' % contrast_id)
save(effects_map, contrast_path)
# note that the effects_map is two-dimensional: 
# these dimensions correspond to 'left' and 'right'

# Create snapshots of the contrasts
vmax = max(- z_map.get_data().min(), z_map.get_data().max())
plot_map(z_map.get_data(), fmri_glm.affine,
         cmap=cm.cold_hot,
         vmin=- vmax,
         vmax=vmax,
         anat=None,
         figure=10,
         threshold=2.5)
plt.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))
plt.show()

print('All the  results were witten in %s' % write_dir)
# Note: fancier visualization of the results are shown
# in the viz3d example
