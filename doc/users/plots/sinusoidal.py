# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This figure is meant to represent a continuous
stimulus having two features, Orientation and Contrast
"""


import pylab
import numpy as np

t = np.linspace(0,10,1000)
o = np.sin(2*np.pi*(t+1)) * np.exp(-t/10)
c = np.sin(2*np.pi*(t+0.2)/4) * np.exp(-t/12)

pylab.plot(t, o, label='Orientation')
pylab.plot(t, c+2.1, label='Contrast')
pylab.legend()

a = pylab.gca()
a.set_yticks([])
a.set_xlabel('Time')
