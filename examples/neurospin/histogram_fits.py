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

import nipy.neurospin.utils.simul_multisubject_fmri_dataset as simul
import nipy.neurospin.utils.emp_null as en

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
gam_gaus_pp = en.Gamma_Gaussian_fit(Beta, Beta)
gam_gaus_pp = np.reshape(gam_gaus_pp, (dimx, dimy, 3))

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

################################################################################
# fit Beta's histogram with a mixture of Gaussians
alpha = 0.01
gaus_mix_pp = en.three_classes_GMM_fit(Beta, None, 
                                       alpha, prior_strength=100)
gaus_mix_pp = np.reshape(gaus_mix_pp, (dimx, dimy, 3))


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

################################################################################
# Fit the null mode of Beta with an empirical normal null

efdr = en.ENN(Beta)
emp_null_fdr = efdr.fdr(Beta)
emp_null_fdr = emp_null_fdr.reshape((dimx, dimy))

pl.subplot(3, 3, 3)
pl.imshow(1-emp_null_fdr, cmap=pl.cm.hot)
pl.colorbar()
pl.title('Empirical FDR\n ')

#efdr.plot()
#pl.title('Empirical FDR fit')

pl.show()
