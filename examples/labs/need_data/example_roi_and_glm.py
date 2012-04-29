# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This is an example where
1. An sequence of fMRI volumes are loaded
2. An ROI mask is loaded
3. A design matrix describing all the effects related to the data is computed
4. A GLM is applied to all voxels in the ROI
5. A summary of the results is provided for certain contrasts
6. A plot of the hrf is provided for the mean reponse in the hrf
7. Fitted/adjusted response plots are provided

Author : Bertrand Thirion, 2010
"""
print __doc__

import numpy as np
import os.path as op
import matplotlib.pylab as mp

from nibabel import save, load, Nifti1Image
from nipy.modalities.fmri.design_matrix import dmtx_light
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.labs.utils.simul_multisubject_fmri_dataset import \
    surrogate_4d_dataset
import get_data_light
import nipy.labs.glm as GLM
import nipy.labs.spatial_models.mroi as mroi
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_image

#######################################
# Simulation parameters
#######################################

# volume mask
mask_path = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 'mask.nii.gz'))
if not op.exists(mask_path):
    get_data_light.get_second_level_dataset()

mask = load(mask_path)

# timing
n_scans = 128
tr = 2.4

# paradigm
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
conditions = np.arange(20) % 2
onsets = np.linspace(5, (n_scans - 1) * tr - 10, 20) # in seconds
hrf_model = 'canonical'
motion = np.cumsum(np.random.randn(n_scans, 6), 0)
add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']

# write directory
swd = '/tmp'

########################################
# Design matrix
########################################

paradigm = np.vstack(([conditions, onsets])).T
paradigm = EventRelatedParadigm(conditions, onsets)
X, names = dmtx_light(frametimes, paradigm, drift_model='cosine', hfcut=128,
                      hrf_model=hrf_model, add_regs=motion,
                      add_reg_names=add_reg_names)


#######################################
# Get the FMRI data
#######################################

fmri_data = surrogate_4d_dataset(mask=mask, dmtx=X, seed=1)[0]

# if you want to save it as an image
# data_file = op.join(swd,'fmri_data.nii')
# save(fmri_data, data_file)

########################################
# Perform a GLM analysis
########################################

# GLM fit
Y = fmri_data.get_data()[mask.get_data() > 0]
model = "ar1"
method = "kalman"
glm = GLM.glm()
glm.fit(Y.T, X, method=method, model=model)

# specifiy the contrast [1 -1 0 ..]
contrast = np.zeros(X.shape[1])
contrast[0] = 1
contrast[1] = -1
my_contrast = glm.contrast(contrast)

# compute the constrast image related to it
zvals = my_contrast.zscore()
zmap = mask.get_data().astype(np.float)
zmap[zmap > 0] = zmap[zmap > 0] * zvals
contrast_image = Nifti1Image(zmap, mask.get_affine())
# if you want to save the contrast as an image
# contrast_path = op.join(swd, 'zmap.nii')
# save(contrast_image, contrast_path)


########################################
# Create ROIs
########################################

positions = np.array([[60, -30, 5], [50, 27, 5]])
# in mm (here in the MNI space)
radii = np.array([8, 6])

domain = grid_domain_from_image(mask)
my_roi = mroi.subdomain_from_balls(domain, positions, radii)

# to save an image of the ROIs
save(my_roi.to_image(), op.join(swd, "roi.nii"))

# exact the time courses with ROIs
thresholded_fmri = fmri_data.get_data()[mask.get_data() > 0]
signal_feature = [thresholded_fmri[my_roi.select_id(id, roi=False)]
                  for id in my_roi.get_id()]
my_roi.set_feature('signal', signal_feature)

# ROI average time courses
my_roi.set_roi_feature('signal_avg', my_roi.representative_feature('signal'))

# roi-level contrast average
thresholded_contrast = contrast_image.get_data()[mask.get_data() > 0]
contrast_feature = [thresholded_contrast[my_roi.select_id(id, roi=False)]
                    for id in my_roi.get_id()]
my_roi.set_feature('contrast', contrast_feature)
my_roi.set_roi_feature('contrast_avg',
                       my_roi.representative_feature('contrast'))

########################################
# GLM analysis on the ROI average time courses
########################################

nreg = len(names)
ROI_tc = my_roi.get_roi_feature('signal_avg')
glm.fit(ROI_tc.T, X, method=method, model=model)

mp.figure()
mp.subplot(1, 2, 1)
b1 = mp.bar(np.arange(nreg - 1), glm.beta[:-1, 0], width=.4, color='blue',
            label='r1')
b2 = mp.bar(np.arange(nreg - 1) + 0.3, glm.beta[:- 1, 1], width=.4,
            color='red', label='r2')
mp.xticks(np.arange(nreg-1), names[:-1])
mp.legend()
mp.title('parameters estimates for the roi time courses')
bx = mp.subplot(1, 2, 2)

my_roi.plot_feature('contrast', bx)
mp.show()


########################################
# fitted and adjusted response
########################################

res = ROI_tc - np.dot(glm.beta.T, X.T)
proj = np.eye(nreg)
proj[2:] = 0
fit = np.dot(np.dot(glm.beta.T, proj), X.T)

# plot it
mp.figure()
for k in range(my_roi.k):
    mp.subplot(my_roi.k, 1, k + 1)
    mp.plot(fit[k])
    mp.plot(fit[k] + res[k], 'r')
    mp.xlabel('time (scans)')
    mp.legend(('effects', 'adjusted'))


###########################################
# hrf for condition 1
############################################

fir_order = 6
X_fir, name_dir = dmtx_light(
    frametimes, paradigm, hrf_model='fir', drift_model='cosine', drift_order=3,
    fir_delays=tr * np.arange(fir_order), add_regs=motion,
    add_reg_names=add_reg_names)
glm.fit(ROI_tc.T, X_fir, method=method, model=model)

mp.figure()
for k in range(my_roi.k):
    mp.subplot(my_roi.k, 1, k + 1)
    var = np.diag(glm.nvbeta[:, :, k]) * glm.s2[k]
    mp.errorbar(np.arange(fir_order), glm.beta[:fir_order, k],
                yerr=np.sqrt(var[:fir_order]))
    mp.errorbar(np.arange(fir_order), glm.beta[fir_order:2 * fir_order, k],
                yerr=np.sqrt(var[fir_order:2 * fir_order]))
    mp.legend(('condition c0', 'condition c1'))
    mp.title('estimated hrf shape')
    mp.xlabel('time(scans)')
mp.show()
