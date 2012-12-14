#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
This is an example where:

1. A sequence of fMRI volumes are loaded
2. An ROI mask is loaded
3. A design matrix describing all the effects related to the data is computed
4. A GLM is applied to all voxels in the ROI
5. A summary of the results is provided for certain contrasts
6. A plot of the HRF is provided for the mean response in the HRF
7. Fitted/adjusted response plots are provided

Needs matplotlib

Author : Bertrand Thirion, 2010
"""
print(__doc__)

from os import mkdir, getcwd, path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nibabel import save, load

from nipy.modalities.fmri.design_matrix import dmtx_light
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.labs.utils.simul_multisubject_fmri_dataset import \
    surrogate_4d_dataset
from nipy.modalities.fmri.glm import GeneralLinearModel
import nipy.labs.spatial_models.mroi as mroi
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_image

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset

#######################################
# Simulation parameters
#######################################

# volume mask
mask_path = path.join(DATA_DIR, 'mask.nii.gz')
if not path.exists(mask_path):
    get_second_level_dataset()

mask = load(mask_path)
mask_array, affine = mask.get_data() > 0, mask.get_affine()

# timing
n_scans = 128
tr = 2.4

# paradigm
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
conditions = np.arange(20) % 2
onsets = np.linspace(5, (n_scans - 1) * tr - 10, 20)  # in seconds
hrf_model = 'canonical'
motion = np.cumsum(np.random.randn(n_scans, 6), 0)
add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']

# write directory
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

########################################
# Design matrix
########################################

paradigm = np.vstack(([conditions, onsets])).T
paradigm = EventRelatedParadigm(conditions, onsets)
X, names = dmtx_light(frametimes, paradigm, drift_model='cosine', hfcut=128,
                      hrf_model=hrf_model, add_regs=motion,
                      add_reg_names=add_reg_names)

########################################
# Create ROIs
########################################

positions = np.array([[60, -30, 5], [50, 27, 5]])
# in mm (here in the MNI space)
radii = np.array([8, 6])

domain = grid_domain_from_image(mask)
my_roi = mroi.subdomain_from_balls(domain, positions, radii)

# to save an image of the ROIs
save(my_roi.to_image(), path.join(write_dir, "roi.nii"))

#######################################
# Get the FMRI data
#######################################
fmri_data = surrogate_4d_dataset(mask=mask, dmtx=X)[0]
Y = fmri_data.get_data()[mask_array]

# artificially added signal in ROIs to make the example more meaningful
activation = 30 * (X.T[1] + .5 * X.T[0])
for (position, radius) in zip(positions, radii):
    Y[((domain.coord - position) ** 2).sum(1) < radius ** 2 + 1] += activation

########################################
# Perform a GLM analysis
########################################

# GLM fit
glm = GeneralLinearModel(X)
glm.fit(Y.T)

# specifiy the contrast [1 -1 0 ..]
contrast = np.hstack((1, -1, np.zeros(X.shape[1] - 2)))

# compute the constrast image related to it
zvals = glm.contrast(contrast).z_score()

########################################
# ROI-based analysis
########################################

# exact the time courses with ROIs
signal_feature = [Y[my_roi.select_id(id, roi=False)] for id in my_roi.get_id()]
my_roi.set_feature('signal', signal_feature)

# ROI average time courses
my_roi.set_roi_feature('signal_avg', my_roi.representative_feature('signal'))

# roi-level contrast average
contrast_feature = [zvals[my_roi.select_id(id, roi=False)]
                    for id in my_roi.get_id()]
my_roi.set_feature('contrast', contrast_feature)
my_roi.set_roi_feature('contrast_avg',
                       my_roi.representative_feature('contrast'))

########################################
# GLM analysis on the ROI average time courses
########################################

n_reg = len(names)
roi_tc = my_roi.get_roi_feature('signal_avg')
glm.fit(roi_tc.T)

plt.figure()
plt.subplot(1, 2, 1)
betas = glm.get_beta()
b1 = plt.bar(np.arange(n_reg - 1), betas[:-1, 0], width=.4, color='blue',
            label='region 1')
b2 = plt.bar(np.arange(n_reg - 1) + 0.3, betas[:- 1, 1], width=.4,
            color='red', label='region 2')
plt.xticks(np.arange(n_reg - 1), names[:-1], fontsize=10)
plt.legend()
plt.title('Parameter estimates \n for the roi time courses')

bx = plt.subplot(1, 2, 2)
my_roi.plot_feature('contrast', bx)

########################################
# fitted and adjusted response
########################################

res = np.hstack([x.resid for x in glm.results_.values()]).T
betas = np.hstack([x.theta for x in glm.results_.values()])
proj = np.eye(n_reg)
proj[2:] = 0
fit = np.dot(np.dot(betas.T, proj), X.T)

# plot it
plt.figure()
for k in range(my_roi.k):
    plt.subplot(my_roi.k, 1, k + 1)
    plt.plot(fit[k])
    plt.plot(fit[k] + res[k], 'r')
    plt.xlabel('time (scans)')
    plt.legend(('effects', 'adjusted'))

###########################################
# hrf for condition 1
############################################

fir_order = 6
X_fir, _ = dmtx_light(
    frametimes, paradigm, hrf_model='fir', drift_model='cosine',
    drift_order=3, fir_delays=np.arange(fir_order), add_regs=motion,
    add_reg_names=add_reg_names)
glm_fir = GeneralLinearModel(X_fir)
plt.figure()

for k in range(my_roi.k):
    # fit a glm on the ROI's time course
    glm_fir.fit(roi_tc[k])
    # access to the corresponding result structure
    res = list(glm_fir.results_.values())[0] # only one value in this case
    plt.subplot(1, my_roi.k, k + 1)

    # get the confidence intervals for the effects and plot them -condition 0
    conf_int = res.conf_int(cols=np.arange(fir_order)).squeeze()
    yerr = (conf_int[:, 1] - conf_int[:, 0]) / 2
    plt.errorbar(np.arange(fir_order), conf_int.mean(1), yerr=yerr)

    # get the confidence intervals for the effects and plot them -condition 1
    conf_int = res.conf_int(cols=np.arange(fir_order, 2 * fir_order)).squeeze()
    yerr = (conf_int[:, 1] - conf_int[:, 0]) / 2
    plt.errorbar(np.arange(fir_order), conf_int.mean(1), yerr=yerr)
    plt.legend(('condition c0', 'condition c1'))
    plt.title('estimated hrf shape')
    plt.xlabel('time(scans)')

plt.show()
