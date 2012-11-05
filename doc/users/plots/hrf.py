# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Plot of the canonical Glover HRF
"""

import numpy as np

from nipy.modalities.fmri import hrf, utils

import matplotlib.pyplot as plt

# hrf.glover is a symbolic function; get a function of time to work on arrays
hrf_func = utils.lambdify_t(hrf.glover(utils.T))

t = np.linspace(0,25,200)
plt.plot(t, hrf_func(t))
a=plt.gca()
a.set_xlabel(r'$t$')
a.set_ylabel(r'$h_{can}(t)$')
