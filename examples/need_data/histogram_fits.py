#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
"""
Example of a script that perfoms histogram analysis of an activation image.
This is based on a real fMRI image.

Simply modify the input image path to make it work on your preferred image.

Needs matplotlib

Author : Bertrand Thirion, 2008-2009
"""

import os

import numpy as np

import scipy.stats as st

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nibabel import load

import nipy.algorithms.statistics.empirical_pvalue as en

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset


# parameters
verbose = 1
theta = float(st.t.isf(0.01, 100))

# paths
mask_image = os.path.join(DATA_DIR, 'mask.nii.gz')
input_image = os.path.join(DATA_DIR, 'spmT_0029.nii.gz')
if (not os.path.exists(mask_image)) or (not os.path.exists(input_image)):
    get_second_level_dataset()

# Read the mask
nim = load(mask_image)
mask = nim.get_data()

# read the functional image
rbeta = load(input_image)
beta = rbeta.get_data()
beta = beta[mask > 0]

mf = plt.figure(figsize=(13, 5))
a1 = plt.subplot(1, 3, 1)
a2 = plt.subplot(1, 3, 2)
a3 = plt.subplot(1, 3, 3)

# fit beta's histogram with a Gamma-Gaussian mixture
bfm = np.array([2.5, 3.0, 3.5, 4.0, 4.5])
bfp = en.gamma_gaussian_fit(beta, bfm, verbose=1, mpaxes=a1)

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
plt.show()
