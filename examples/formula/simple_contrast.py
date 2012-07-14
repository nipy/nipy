# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" A simple contrast for an FMRI HRF model """

import numpy as np

from nipy.algorithms.statistics.api import Formula, make_recarray
from nipy.modalities.fmri import utils, hrf
from nipy.modalities.fmri.fmristat import hrf as delay

# We take event onsets, and a specified HRF model, and make symbolic functions
# of time
c1 = utils.events([3,7,10], f=hrf.glover) # Symbolic function of time
c2 = utils.events([1,3,9], f=hrf.glover) # Symbolic function of time
c3 = utils.events([3,4,6], f=delay.spectral[0]) # Symbolic function of time

# We can also use a Fourier basis for some other onsets - again making symbolic
# functions of time
d = utils.fourier_basis([3,5,7]) # Formula

# Make a formula for all four sets of onsets
f = Formula([c1,c2,c3]) + d

# A contrast is a formula expressed on the elements of the design formula
contrast = Formula([c1-c2, c1-c3])

# Instantiate actual values of time at which to create the design matrix rows
t = make_recarray(np.linspace(0,20,50), 't')

# Make the design matrix, and get contrast matrices for the design
X, c = f.design(t, return_float=True, contrasts={'C':contrast})

# c is a dictionary, containing a 2 by 9 matrix - the F contrast matrix for our
# contrast of interest
assert X.shape == (50, 9)
assert c['C'].shape == (2, 9)

# In this case the contrast matrix is rather obvious.
np.testing.assert_almost_equal(c['C'],
                               [[1,-1, 0, 0, 0, 0, 0, 0, 0],
                                [1, 0, -1, 0, 0, 0, 0, 0, 0]])

# We can get the design implied by our contrast at our chosen times
preC = contrast.design(t, return_float=True)
np.testing.assert_almost_equal(preC[:, 0], X[:, 0] - X[:, 1])
np.testing.assert_almost_equal(preC[:, 1], X[:, 0] - X[:, 2])

# So, X . c['C'].T \approx preC
np.testing.assert_almost_equal(np.dot(X, c['C'].T), preC)

# So what is the matrix C such that preC = X . C?  Yes, it's c['C']
C = np.dot(np.linalg.pinv(X), preC).T
np.testing.assert_almost_equal(C, c['C'])

# The contrast matrix (approx equal to c['C'])
print C
