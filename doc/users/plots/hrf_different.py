"""
This example uses a different HRF for different event types
"""

import pylab
import numpy as np
from sympy import lambdify
from neuroimaging.modalities.fmri import hrf

glover = hrf.glover_sympy
afni = hrf.afni_sympy

ta = [0,4,8,12,16]; tb = [2,6,10,14,18]
ba = 1; bb = -2
na = ba * sum([glover.subs(hrf.t, hrf.t - t) for t in ta])
nb = bb * sum([afni.subs(hrf.t, hrf.t - t) for t in tb])

nav = lambdify(hrf.vector_t, na.subs(hrf.t, hrf.vector_t), 'numpy')
nbv = lambdify(hrf.vector_t, nb.subs(hrf.t, hrf.vector_t), 'numpy')

t = np.linspace(0,30,200)
pylab.plot(t, nav(t), c='r', label='Face')
pylab.plot(t, nbv(t), c='b', label='Object')
pylab.plot(t, nbv(t)+nav(t), c='g', label='Neuronal')

for t in ta:
    pylab.plot([t,t],[0,ba*0.5],c='r')
for t in tb:
    pylab.plot([t,t],[0,bb*0.5],c='b')
pylab.plot([0,30], [0,0],c='#000000')
pylab.legend()

pylab.show()
