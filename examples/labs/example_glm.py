# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This is an example where:

1. An sequence of fMRI volumes are simulated
2. A design matrix describing all the effects related to the data is computed
3. A GLM is applied to all voxels
4. A contrast image is created

Requires matplotlib

Author : Bertrand Thirion, 2010
"""
print __doc__

import os
import os.path as op

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nibabel import save, Nifti1Image

import nipy.modalities.fmri.design_matrix as dm
from nipy.labs.utils.simul_multisubject_fmri_dataset import \
     surrogate_4d_dataset
from nipy.modalities.fmri.glm import GeneralLinearModel
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm

#######################################
# Simulation parameters
#######################################

# volume mask
shape = (20, 20, 20)
affine = np.eye(4)

# Acquisition parameters: number of scans (n_scans) and volume repetition time
# value in seconds
n_scans = 128
tr = 2.4

# input paradigm information
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)

# conditions are 0 1 0 1 0 1 ...
conditions = np.arange(20) % 2

# 20 onsets (in sec), first event 10 sec after the start of the first scan
onsets = np.linspace(5, (n_scans - 1) * tr - 10, 20)

# model with canonical HRF (could also be :
#   'canonical with derivative' or 'fir'
hrf_model = 'canonical'

# fake motion parameters to be included in the model
motion = np.cumsum(np.random.randn(n_scans, 6), 0)
add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']

########################################
# Design matrix
########################################

paradigm = EventRelatedParadigm(conditions, onsets)
X, names = dm.dmtx_light(frametimes, paradigm, drift_model='cosine',
                         hfcut=128, hrf_model=hrf_model, add_regs=motion,
                         add_reg_names=add_reg_names)


#######################################
# Get the FMRI data
#######################################

fmri_data = surrogate_4d_dataset(shape=shape, n_scans=n_scans)[0]

# if you want to save it as an image
data_file = 'fmri_data.nii'
save(fmri_data, data_file)

########################################
# Perform a GLM analysis
########################################

# GLM fit
Y = fmri_data.get_data().reshape(np.prod(shape), n_scans)
glm = GeneralLinearModel(X)
glm.fit(Y.T)

# specify the contrast [1 -1 0 ..]
contrast = np.zeros(X.shape[1])
contrast[0] = 1
contrast[1] = - 1

# compute the constrast image related to it
zvals = glm.contrast(contrast).z_score()
contrast_image = Nifti1Image(np.reshape(zvals, shape), affine)

# if you want to save the contrast as an image
contrast_path = 'zmap.nii'
save(contrast_image, contrast_path)

print ('Wrote the some of the results as images in directory %s' %
       op.abspath(os.getcwd()))

h, c = np.histogram(zvals, 100)

# Show the histogram
plt.figure()
plt.bar(c[: - 1], h, width=.1)
plt.title(' Histogram of the z-values')
plt.show()
