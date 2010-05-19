# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This figure is meant to represent the neuronal block model
with Faces at times [0,4,8,12,16] and Objects presented at [2,6,10,14,18]
each presented for 0.5 seconds
and a coefficient of +1 for Faces, -2 for Objects.

"""

import pylab
import numpy as np


from sympy import Symbol, Piecewise, lambdify
ta = [0,4,8,12,16]; tb = [2,6,10,14,18]
ba = Symbol('ba'); bb = Symbol('bb'); t = Symbol('t')
fa = sum([Piecewise((0, (t<_t)), ((t-_t)/0.5, (t<_t+0.5)), (1, (t >= _t+0.5))) for _t in ta])*ba
fb = sum([Piecewise((0, (t<_t)), ((t-_t)/0.5, (t<_t+0.5)), (1, (t >= _t+0.5))) for _t in tb])*bb
N = fa+fb

Nn = N.subs(ba,1)
Nn = Nn.subs(bb,-2)

NNl = lambdify(t, Nn)

tt = np.linspace(-1,21,121)
pylab.plot(tt, [NNl(float(_t)) for _t in tt])

a = pylab.gca()
a.set_ylim([-5.5,1.5])
a.set_ylabel('Neuronal (cumulative)')
a.set_xlabel('Time')

pylab.show()
