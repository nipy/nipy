# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This plot demonstrates a neuronal model that is a sum
of delta functions times coefficient values
"""

import pylab
from matplotlib import rc
rc('text', usetex=True)

ba = 1
bb = -2

ta = [0,4,8,12,16]; tb = [2,6,10,14,18]

for t in ta:
    pylab.plot([t,t],[0,ba],c='r')
for t in tb:
    pylab.plot([t,t],[0,bb],c='b')

a = pylab.gca()
a.set_xlabel(r'$t$')
a.set_ylabel(r'$n(t)$')

