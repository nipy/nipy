# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This example simulates a number of pure Gaussian white noise signals, then fits
each one in terms of two regressors: a constant baseline, and a linear function
of time. The voxelwise t statistics associated with the baseline coefficient are
then computed.
"""
print __doc__

import numpy as np

from nipy.modalities.fmri.glm import GeneralLinearModel

dimt = 100
dimx = 10
dimy = 11
dimz = 12

# axis defines the "time direction"
y = np.random.randn(dimt, dimx * dimy * dimz)
axis = 0

X = np.array([np.ones(dimt), range(dimt)])
X = X.T ## the design matrix X must have dimt lines

mod = GeneralLinearModel(X)
mod.fit(y)

# Define a t contrast
tcon = mod.contrast([1, 0])

# Compute the t-stat
t = tcon.stat()
## t = tcon.stat(baseline=1) to test effects > 1

# Compute the p-value
p = tcon.p_value()

# Compute the z-score
z = tcon.z_score()

# Perform a F test without keeping the F stat
p = mod.contrast([[1, 0], [1, - 1]]).p_value()

print np.shape(y)
print np.shape(X)
print np.shape(z)
