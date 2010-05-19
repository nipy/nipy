# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import pylab

from nipy.modalities.fmri.utils import events, Symbol, Vectorize
from nipy.modalities.fmri.hrf import glover_sympy

a = Symbol('a')
b = np.linspace(0,50,6)
ba = b*([-1,1]*3)
d = events(b, amplitudes=ba, g=a+0.5*a**2, f=glover_sympy)
dt = Vectorize(d)
tt = np.linspace(0,60,601)

pylab.plot(tt, dt(tt), c='r')
for bb, aa in zip(b,ba):
    pylab.plot([bb,bb],[0,25*aa], c='b')

pylab.show()
