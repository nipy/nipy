# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This example uses a different HRF for different event types
"""

import numpy as np

import matplotlib.pyplot as plt

from nipy.modalities.fmri import hrf
from nipy.modalities.fmri.utils import T, lambdify_t


# HRFs as functions of (symbolic) time
glover = hrf.glover(T)
afni = hrf.afni(T)

ta = [0,4,8,12,16]; tb = [2,6,10,14,18]
ba = 1; bb = -2
na = ba * sum([glover.subs(T, T - t) for t in ta])
nb = bb * sum([afni.subs(T, T - t) for t in tb])

nav = lambdify_t(na)
nbv = lambdify_t(nb)

t = np.linspace(0,30,200)
plt.plot(t, nav(t), c='r', label='Face')
plt.plot(t, nbv(t), c='b', label='Object')
plt.plot(t, nbv(t)+nav(t), c='g', label='Combined')

for t in ta:
    plt.plot([t,t],[0,ba*0.5],c='r')
for t in tb:
    plt.plot([t,t],[0,bb*0.5],c='b')
plt.plot([0,30], [0,0],c='#000000')
plt.legend()

plt.show()
