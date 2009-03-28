"""
Example of a script that perfoms histogram analysis of an activation
image, to estimate activation Z-score with various heuristics:

   * Gamma-Gaussian model
   * Gaussian mixture model
   * Empirical normal null

This example is based on a (simplistic) simulated image.

"""
# Author : Bertrand Thirion, 2008-2009

import numpy as np
import scipy.stats as st
import os.path as op
import fff2.spatial_models.bayesian_structural_analysis as bsa
import fff2.utils.simul_2d_multisubject_fmri_dataset as simul
from fff2.utils.zscore import zscore

################################################################################
# simulate the data
dimx = 60
dimy = 60
pos = 2*np.array([[6,7],[10,10],[15,10]])
ampli = np.array([3,4,4])

dataset = simul.make_surrogate_array(nbsubj=1, dimx=dimx, dimy=dimy, pos=pos,
                                     ampli=ampli, width=10.0).squeeze()

import pylab as pl
fig = pl.figure(figsize=(12, 10))
pl.subplot(3, 3, 1)
pl.imshow(dataset, cmap=pl.cm.hot)
pl.colorbar()
pl.title('Raw data')

Beta = dataset.ravel().squeeze()

################################################################################
# fit Beta's histogram with a Gamma-Gaussian mixture
gam_gaus_zscore = zscore(bsa._GGM_priors_(Beta, Beta))
gam_gaus_zscore = np.reshape(gam_gaus_zscore, (dimx, dimy, 3))

pl.figure(fig.number)
pl.subplot(3, 3, 4)
pl.imshow(gam_gaus_zscore[..., 0], cmap=pl.cm.hot)
pl.title('Gamme-Gaussian mixture,\n first component Z-score')
pl.colorbar()
pl.subplot(3, 3, 5)
pl.imshow(gam_gaus_zscore[..., 1], cmap=pl.cm.hot)
pl.title('Gamme-Gaussian mixture,\n second component Z-score')
pl.colorbar()
pl.subplot(3, 3, 6)
pl.imshow(gam_gaus_zscore[..., 2], cmap=pl.cm.hot)
pl.title('Gamme-Gaussian mixture,\n third component Z-score')
pl.colorbar()

################################################################################
# fit Beta's histogram with a mixture of Gaussians
alpha = 0.01
theta = float(st.t.isf(0.01, 100))
# FIXME: Ugly crasher if the second Beta is not reshaped
gaus_mix_zscore = zscore(bsa._GMM_priors_(Beta, Beta.reshape(-1, 1), theta, 
                            alpha, 
                            prior_strength=100))
gaus_mix_zscore = np.reshape(gaus_mix_zscore, (dimx, dimy, 3))

pl.figure(fig.number)
pl.subplot(3, 3, 7)
pl.imshow(gaus_mix_zscore[..., 0], cmap=pl.cm.hot)
pl.title('Gaussian mixture,\n first component Z-score')
pl.colorbar()
pl.subplot(3, 3, 8)
pl.imshow(gaus_mix_zscore[..., 1], cmap=pl.cm.hot)
pl.title('Gaussian mixture,\n second component Z-score')
pl.colorbar()
pl.subplot(3, 3, 9)
pl.imshow(gaus_mix_zscore[..., 2], cmap=pl.cm.hot)
pl.title('Gamme-Gaussian mixture,\n third component Z-score')
pl.colorbar()

################################################################################
# Fit the null mode of Beta with an empirical normal null
import fff2.utils.emp_null as en
efdr = en.ENN(Beta)
emp_null_zcore = zscore(efdr.fdr(Beta))
emp_null_zcore = emp_null_zcore.reshape((dimx, dimy))

pl.subplot(3, 3, 3)
pl.imshow(emp_null_zcore, cmap=pl.cm.hot)
pl.colorbar()
pl.title('Empirical normal null\n Z-score')

efdr.plot()
pl.title('Empirical normal null fit')

pl.show()
