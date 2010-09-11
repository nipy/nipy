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
import nipy.neurospin.utils.emp_null as en
import get_data_light

from nipy.io.imageformats import load

# parameters
verbose = 1
theta = float(st.t.isf(0.01,100))

# paths
data_dir = get_data_light.get_it()
mask_image = os.path.join(data_dir, 'mask.nii.gz')
input_image = os.path.join(data_dir, 'spmT_0029.nii.gz')

# Read the mask
nim = load(mask_image)
mask = nim.get_data()

# read the functional image
rbeta = load(input_image)
beta = rbeta.get_data()
beta = beta[mask>0]

mf = mp.figure()
a1 = mp.subplot(1,3,1)
a2 = mp.subplot(1,3,2)
a3 = mp.subplot(1,3,3)

# fit beta's histogram with a Gamma-Gaussian mixture
bfm = np.array([2.5,3.0,3.5,4.0,4.5])
bfp = en.Gamma_Gaussian_fit(beta, bfm, verbose=2, mpaxes=a1)

# fit beta's histogram with a mixture of Gaussians
alpha = 0.01
pstrength = 100
bfq = en.three_classes_GMM_fit(beta, bfm, alpha, pstrength,
                               verbose=2, mpaxes=a2)

# fit the null mode of beta with the robust method
efdr = en.ENN(beta)
efdr.learn()
efdr.plot(bar=0,mpaxes=a3)

mf.set_size_inches(15, 5, forward=True)
mp.show()
