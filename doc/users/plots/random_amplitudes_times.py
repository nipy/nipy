# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This figure is meant to represent an event-type design with
events at random times centered [0,2,4,6,8] and random
amplitudes centered [0,1.1,2.3,0.9,0.3].
"""

import pylab, matplotlib
import numpy as np


for t, y in zip([0,4,8,12,16], [0,1.1,2.3,0.9,0.3]):
    dt = np.array([-0.5,0.5,0.5,-0.5])
    dy = np.array([-0.1,-0.1,0.1,0.1])
    pylab.fill(t+dt,y+dy, 'r')

a = pylab.gca()
a.set_yticks([0,2])
a.set_xlim([-1,18])
a.set_xlabel('Time')
a.set_ylabel('Amplitude')

