#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
Full step-by-step example of fitting a GLM to experimental data and visualizing
the results.

More specifically:

1. A sequence of fMRI volumes are loaded
2. A design matrix describing all the effects related to the data is computed
3. a mask of the useful brain volume is computed
4. A GLM is applied to the dataset (effect/covariance,
   then contrast estimation)

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
hrf_model = 'canonical with derivative'
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

paradigm = load_paradigm_from_csv_file(paradigm_file)['0']

design_matrix = make_dmtx(frametimes, paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')

plt.savefig(path.join(write_dir, 'design_matrix.png'))

#########################################
# Specify the contrasts
#########################################

# simplest ones
contrasts = {}
n_columns = len(design_matrix.names)
for i in range(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

# and more complex/ interesting ones
contrasts["audio"] = contrasts["clicDaudio"] + contrasts["clicGaudio"] +\
                     contrasts["calculaudio"] + contrasts["phraseaudio"]
contrasts["video"] = contrasts["clicDvideo"] + contrasts["clicGvideo"] + \
                     contrasts["calculvideo"] + contrasts["phrasevideo"]
contrasts["left"] = contrasts["clicGaudio"] + contrasts["clicGvideo"]
contrasts["right"] = contrasts["clicDaudio"] + contrasts["clicDvideo"]
contrasts["computation"] = contrasts["calculaudio"] + contrasts["calculvideo"]
contrasts["sentences"] = contrasts["phraseaudio"] + contrasts["phrasevideo"]
contrasts["H-V"] = contrasts["damier_H"] - contrasts["damier_V"]
contrasts["V-H"] = contrasts["damier_V"] - contrasts["damier_H"]
contrasts["left-right"] = contrasts["left"] - contrasts["right"]
contrasts["right-left"] = contrasts["right"] - contrasts["left"]
contrasts["audio-video"] = contrasts["audio"] - contrasts["video"]
contrasts["video-audio"] = contrasts["video"] - contrasts["audio"]
contrasts["computation-sentences"] = contrasts["computation"] -  \
                                     contrasts["sentences"]
contrasts["reading-visual"] = contrasts["sentences"] * 2 - \
                              contrasts["damier_H"] - contrasts["damier_V"]
contrasts['effects_of_interest'] = np.eye(25)[:20:2]

########################################
# Perform a GLM analysis
########################################

print('Fitting a GLM (this takes time)...')
fmri_glm = FMRILinearModel(data_path, design_matrix.matrix,
                           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

#########################################
# Estimate the contrasts
#########################################

print('Computing contrasts...')
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    # save the z_image
    image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
    z_map, = fmri_glm.contrast(contrast_val, con_id=contrast_id, output_z=True)
    save(z_map, image_path)

    # Create snapshots of the contrasts
    vmax = max(- z_map.get_data().min(), z_map.get_data().max())
    if index > 0:
        plt.clf()
    plot_map(z_map.get_data(), z_map.get_affine(),
             cmap=cm.cold_hot,
             vmin=- vmax,
             vmax=vmax,
             anat=None,
             figure=10,
             threshold=2.5)
    plt.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

print("All the  results were witten in %s" % write_dir)

plt.show()
