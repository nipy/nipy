# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This figure is meant to represent the neuronal event-related model

and a coefficient of +1 for Faces, -2 for Objects.

"""

import pylab
import numpy as np


from sympy import Symbol, Heaviside, lambdify
ta = [0,4,8,12,16]; tb = [2,6,10,14,18]
ba = Symbol('ba'); bb = Symbol('bb'); t = Symbol('t')
fa = sum([Heaviside(t-_t) for _t in ta]) * ba
fb = sum([Heaviside(t-_t) for _t in tb]) * bb
N = fa+fb

Nn = N.subs(ba,1)
Nn = Nn.subs(bb,-2)

Nn = lambdify(t, Nn)

tt = np.linspace(-1,21,1201)
pylab.step(tt, [Nn(_t) for _t in tt])

a = pylab.gca()
a.set_ylim([-5.5,1.5])
a.set_ylabel('Neuronal (cumulative)')
a.set_xlabel('Time')

pylab.show()
