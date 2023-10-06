# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This figure is meant to represent the neuronal event-related model and a
coefficient of +1 for Faces, -2 for Objects.
"""

import matplotlib.pyplot as plt
import numpy as np
from sympy import Heaviside, Symbol, lambdify

ta = [0, 4, 8, 12, 16]
tb = [2, 6, 10, 14, 18]
ba = Symbol('ba')
bb = Symbol('bb')
t = Symbol('t')
fa = sum([Heaviside(t - _t) for _t in ta]) * ba
fb = sum([Heaviside(t - _t) for _t in tb]) * bb
N = fa + fb

Nn = N.subs(ba, 1)
Nn = Nn.subs(bb, -2)

# Use Numpy heaviside for lambdify, with y=1 for x=0.
modules = [{'Heaviside': lambda x, y: np.heaviside(x, 1)}, 'numpy']
neuronal_func = lambdify(t, Nn, modules=modules)

tt = np.linspace(-1, 21, 1201)
neuronal = neuronal_func(tt)

plt.step(tt, neuronal)

a = plt.gca()
a.set_ylim([-5.5, 1.5])
a.set_ylabel('Neuronal (cumulative)')
a.set_xlabel('Time')

plt.show()
