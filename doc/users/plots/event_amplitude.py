# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import pylab

from nipy.modalities.fmri.utils import events, Symbol, lambdify_t
from nipy.modalities.fmri.hrf import glover

# Symbol for amplitude
a = Symbol('a')

# Some event onsets regularly spaced
onsets = np.linspace(0,50,6)

# Make amplitudes from onset times (greater as function of time)
amplitudes = onsets[:]

# Flip even numbered amplitudes
amplitudes = amplitudes * ([-1, 1] * 3)

# Make event functions
evs = events(onsets, amplitudes=amplitudes, g=a + 0.5 * a**2, f=glover)

# Real valued function for symbolic events
real_evs = lambdify_t(evs)

# Time points at which to sample
t_samples = np.linspace(0,60,601)

pylab.plot(t_samples, real_evs(t_samples), c='r')
for onset, amplitude in zip(onsets, amplitudes):
    pylab.plot([onset, onset],[0, 25 * amplitude], c='b')

pylab.show()
