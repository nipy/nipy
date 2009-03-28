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
import fff2.spatial_models.bayesian_structural_analysis as bsa
import nifti

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
mask = np.transpose(nim.asarray())
xyz = np.array(np.where(mask))
nbvox = np.size(xyz,1)

# read the functional images
Beta = []
rbeta = nifti.NiftiImage(betas[s][0])
beta = np.transpose(rbeta.asarray())
beta = beta[mask>0]
Beta.append(beta)
Beta = np.transpose(np.array(Beta))

# fit Beta's histogram with a Gamma-Gaussian mixture
Bfm = np.array([2.5,3.0,3.5,4.0,4.5])
Bfp = bsa._GGM_priors_(np.squeeze(Beta),Bfm,verbose=2)

# fit Beta's histogram with a mixture of Gaussians
alpha = 0.01
prior_strength = 100
Bfm = np.reshape(Bfm,(np.size(Bfm),1))
Bfq = bsa._GMM_priors_(np.squeeze(Beta),Bfm,theta,alpha,prior_strength,verbose=2)

# fit the null mode of Beta with the robust method
import fff2.utils.emp_null as en
efdr = en.ENN(Beta)
efdr.learn()
#Bfr = efdr.fdr(Bfm)
efdr.plot(bar=0)
