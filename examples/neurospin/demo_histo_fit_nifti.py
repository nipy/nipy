"""
Example of a script that perfoms histogram analysis of an activation image.
This is based on a real fMRI image

Simply modify the input image path to make it work on your preferred
nifti image

todo:  the 3 figures are ugly: use subfigure instead

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
import os
import nifti
import scipy.stats as st
import nipy.neurospin.utils.emp_null as en
import get_data_light
get_data_light.getIt()

# parameters
verbose = 1
theta = float(st.t.isf(0.01,100))

# paths
data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
MaskImage = os.path.join(data_dir,'mask.nii.gz')
InputImage = os.path.join(data_dir,'spmT_0029.nii.gz')


# Read the referential
nim = nifti.NiftiImage(MaskImage)
ref_dim = nim.getVolumeExtent()
grid_size = np.prod(ref_dim)
sform = nim.header['sform']
voxsize = nim.getVoxDims()

# Read the masks and compute the "intersection"
mask = nim.asarray().T

# read the functional image
rbeta = nifti.NiftiImage(InputImage)
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

import matplotlib.pylab as mp
mp.show()
