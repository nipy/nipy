# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This figure is meant to represent an event-type with Faces presented
at times [0,4,8,12,16] and Objects presented at [2,6,10,14,18].

There are two values for Y: one for 'Face' and one for 'Object'
"""

import pylab
import numpy as np

for t in [0,4,8,12,16]:
    pylab.plot([t,t+0.5], [1,1], c='r', label='Face', linewidth=3)
for t in [2,6,10,14,18]:
    pylab.plot([t,t+0.5], [0,0], c='b', label='Object', linewidth=3)


a = pylab.gca()
a.set_ylim([-0.1,1.1])
a.set_yticks([0,1])
a.set_yticklabels(['Object', 'Face'])
a.set_xlim([-0.5,10])
a.set_xlabel('Time')

