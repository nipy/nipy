"""
Example of a script that perfoms histogram analysis of an activation image.
This is based on a real fMRI image

Simply modify the input image path to make it work on your preferred
nifti image

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
import scipy.stats as st
import os.path as op
import nifti

import nipy.neurospin.utils.emp_null as en

nbru = range(1,13)

nbeta = [29]
theta = float(st.t.isf(0.01,100))
verbose = 1

swd = "/tmp/"

# a mask of the brain in each subject
Mask_Images =["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/mask.img" % bru for bru in nbru]

# activation image in each subject
betas = [["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/spmT_%04d.img" % (bru, n) for n in nbeta] for bru in nbru]

s=6

# Read the referential
nim = nifti.NiftiImage(Mask_Images[s])
ref_dim = nim.getVolumeExtent()
grid_size = np.prod(ref_dim)
sform = nim.header['sform']
voxsize = nim.getVoxDims()

# Read the masks and compute the "intersection"
mask = nim.asarray().T
xyz = np.array(np.where(mask))
nbvox = np.size(xyz,1)

# read the functional image
rbeta = nifti.NiftiImage(betas[s][0])
beta = rbeta.asarray().T
beta = beta[mask>0]

# fit beta's histogram with a Gamma-Gaussian mixture
bfm = np.array([2.5,3.0,3.5,4.0,4.5])
bfp = en.Gamma_Gaussian_fit(np.squeeze(beta),bfm,verbose=2)

# fit beta's histogram with a mixture of Gaussians
alpha = 0.01
prior_strength = 100
bfq = en.three_classes_GMM_fit(beta, bfm, alpha, prior_strength,verbose=2)

# fit the null mode of beta with the robust method
efdr = en.ENN(beta)
efdr.learn()
efdr.plot(bar=0)
