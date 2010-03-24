"""
This is an example where
1. An sequence of fMRI volumes are loaded
2. An ROI mask is loaded
3. A design matrix describing all the effects related to the data is computed
4. A GLM is applied to all voxels in the ROI
5. A summary of the results is provided for certain contrasts
6. A plot of the hrf is provided for the mean reposne in the hrf
7. Fitted/adjusted response plots are provided
"""

import numpy as np
import os.path as op
import matplotlib.pylab as mp

from nipy.io.imageformats import load, save, Nifti1Image
from nipy.neurospin.utils.design_matrix import dmtx_light
from nipy.neurospin.utils.simul_multisubject_fmri_dataset import surrogate_4d_dataset
import get_data_light
import nipy.neurospin.glm as GLM
from nipy.neurospin.utils.roi import MultipleROI

#######################################
# Simulation parameters
#######################################

# volume mask
get_data_light.getIt()
mask_path = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 'mask.nii.gz'))
mask = load(mask_path)

# timing
n_scans  =128
tr = 2.4

# paradigm
frametimes = np.linspace(0, (n_scans-1)*tr, n_scans)
conditions = np.arange(20)%2
onsets = np.linspace(5, (n_scans-1)*tr-10, 20) # in seconds
hrf_model = 'Canonical'
motion = np.cumsum(np.random.randn(n_scans, 6),0)
add_reg_names = ['tx','ty','tz','rx','ry','rz']

# write directory
swd = '/tmp'

########################################
# Design matrix
########################################

paradigm = np.vstack(([conditions, onsets])).T
X, names = dmtx_light(frametimes, paradigm, drift_model='Cosine', hfcut=128,
                      hrf_model=hrf_model, add_regs=motion,
                      add_reg_names=add_reg_names)
#mp.matshow(X/np.sqrt((X**2).sum(0)))
#mp.show()

#######################################
# Get the FMRI data
#######################################

data_file = op.join(swd,'toto.nii')
data = surrogate_4d_dataset(mask=mask, dmtx=X, seed=1,
                            out_image_file=data_file)

########################################
# Perform a GLM
########################################

Y = data[mask.get_data()>0, :]
model = "ar1"
method = "kalman"
glm = GLM.glm()
glm.fit(Y.T, X, method=method, model=model)

# compute the constrast image related to [1 -1 0 ..]
contrast = np.zeros(X.shape[1])
contrast[0] = 1; contrast[1] = -1
my_contrast = glm.contrast(contrast)
zvals = my_contrast.zscore()
zmap = mask.get_data().astype(np.float)
zmap[zmap>0] = zmap[zmap>0]*zvals
contrast_path = op.join(swd,'zmap.nii')
save(Nifti1Image(zmap, mask.get_affine()), contrast_path)


########################################
# Create ROIs
########################################

positions = np.array([[60, -30, 5],[50, 27, 5]])
# in mm (here in the MNI space)
radii = np.array([8,6])
mroi = MultipleROI( affine=mask.get_affine(), shape=mask.get_shape())
mroi.as_multiple_balls(positions, radii)
mroi.make_image((op.join(swd, "roi.nii")))

# roi time courses
mroi.set_discrete_feature_from_image('activ', data_file)
mroi.discrete_to_roi_features('activ')

# roi-level contrast average
mroi.set_discrete_feature_from_image('contrast', contrast_path)
mroi.discrete_to_roi_features('contrast')
mroi.plot_roi_feature('contrast')
mp.show()









########################################
# GLM analysis
########################################



