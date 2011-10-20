# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
__doc__ = \
"""
Example of a script that perfoms histogram analysis of an activation
image, to estimate activation Z-score with various heuristics:

   * Gamma-Gaussian model
   * Gaussian mixture model
   * Empirical normal null

This example is based on a (simplistic) simulated image.
"""
# Author : Bertrand Thirion, Gael Varoquaux 2008-2009
print __doc__

import numpy as np

import nipy.labs.utils.simul_multisubject_fmri_dataset as simul
import nipy.algorithms.statistics.empirical_pvalue as en

###############################################################################
# simulate the data
shape = (60, 60)
pos = 2 * np.array([[6, 7], [10, 10], [15, 10]])
ampli = np.array([3, 4, 4])

dataset = simul.surrogate_2d_dataset(n_subj=1, shape=shape, pos=pos,
                                     ampli=ampli, width=10.0).squeeze()

import pylab as pl
fig = pl.figure(figsize=(12, 10))
pl.subplot(3, 3, 1)
pl.imshow(dataset, cmap=pl.cm.hot)
pl.colorbar()
pl.title('Raw data')

Beta = dataset.ravel().squeeze()

###############################################################################
# fit Beta's histogram with a Gamma-Gaussian mixture
gam_gaus_pp = en.Gamma_Gaussian_fit(Beta, Beta)
gam_gaus_pp = np.reshape(gam_gaus_pp, (shape[0], shape[1], 3))

pl.figure(fig.number)
pl.subplot(3, 3, 4)
pl.imshow(gam_gaus_pp[..., 0], cmap=pl.cm.hot)
pl.title('Gamma-Gaussian mixture,\n first component posterior proba.')
pl.colorbar()
pl.subplot(3, 3, 5)
pl.imshow(gam_gaus_pp[..., 1], cmap=pl.cm.hot)
pl.title('Gamma-Gaussian mixture,\n second component posterior proba.')
pl.colorbar()
pl.subplot(3, 3, 6)
pl.imshow(gam_gaus_pp[..., 2], cmap=pl.cm.hot)
pl.title('Gamma-Gaussian mixture,\n third component posterior proba.')
pl.colorbar()

###############################################################################
# fit Beta's histogram with a mixture of Gaussians
alpha = 0.01
gaus_mix_pp = en.three_classes_GMM_fit(Beta, None, 
                                       alpha, prior_strength=100)
gaus_mix_pp = np.reshape(gaus_mix_pp, (shape[0], shape[1], 3))


pl.figure(fig.number)
pl.subplot(3, 3, 7)
pl.imshow(gaus_mix_pp[..., 0], cmap=pl.cm.hot)
pl.title('Gaussian mixture,\n first component posterior proba.')
pl.colorbar()
pl.subplot(3, 3, 8)
pl.imshow(gaus_mix_pp[..., 1], cmap=pl.cm.hot)
pl.title('Gaussian mixture,\n second component posterior proba.')
pl.colorbar()
pl.subplot(3, 3, 9)
pl.imshow(gaus_mix_pp[..., 2], cmap=pl.cm.hot)
pl.title('Gamma-Gaussian mixture,\n third component posterior proba.')
pl.colorbar()

###############################################################################
# Fit the null mode of Beta with an empirical normal null

efdr = en.NormalEmpiricalNull(Beta)
emp_null_fdr = efdr.fdr(Beta)
emp_null_fdr = emp_null_fdr.reshape(shape)

pl.subplot(3, 3, 3)
pl.imshow(1 - emp_null_fdr, cmap=pl.cm.hot)
pl.colorbar()
pl.title('Empirical FDR\n ')
pl.show()
