# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of a script that perfoms histogram analysis of an activation image.
This is based on a real fMRI image

Simply modify the input image path to make it work on your preferred
image

Author : Bertrand Thirion, 2008-2009
"""

import os
import numpy as np
import matplotlib.pylab as mp
import scipy.stats as st

from nibabel import load
import nipy.algorithms.statistics.empirical_pvalue as en
import get_data_light


# parameters
verbose = 1
theta = float(st.t.isf(0.01, 100))

# paths
data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
mask_image = os.path.join(data_dir, 'mask.nii.gz')
input_image = os.path.join(data_dir, 'spmT_0029.nii.gz')
if (not os.path.exists(mask_image)) or (not os.path.exists(input_image)):
    get_data_light.get_second_level_dataset()

# Read the mask
nim = load(mask_image)
mask = nim.get_data()

# read the functional image
rbeta = load(input_image)
beta = rbeta.get_data()
beta = beta[mask > 0]

mf = mp.figure(figsize=(13, 5))
a1 = mp.subplot(1, 3, 1)
a2 = mp.subplot(1, 3, 2)
a3 = mp.subplot(1, 3, 3)

# fit beta's histogram with a Gamma-Gaussian mixture
bfm = np.array([2.5, 3.0, 3.5, 4.0, 4.5])
bfp = en.Gamma_Gaussian_fit(beta, bfm, verbose=1, mpaxes=a1)

# fit beta's histogram with a mixture of Gaussians
alpha = 0.01
pstrength = 100
bfq = en.three_classes_GMM_fit(beta, bfm, alpha, pstrength,
                               verbose=1, mpaxes=a2)

# fit the null mode of beta with the robust method
efdr = en.NormalEmpiricalNull(beta)
efdr.learn()
efdr.plot(bar=0, mpaxes=a3)

a1.set_title('Fit of the density with \n a Gamma-Gaussian mixture')
a2.set_title('Fit of the density with \n a mixture of Gaussians')
a3.set_title('Robust fit of the density \n with a single Gaussian')
mp.show()
