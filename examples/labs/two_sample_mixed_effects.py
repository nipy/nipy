# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Demo two sample mixed effect models

Needs matplotlib
"""
print __doc__

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nipy.labs.group import twosample

n1 = 8
n2 = 8

y1 = np.random.rand(n1)
v1 = .1 * np.random.rand(n1)

y2 = np.random.rand(n2)
v2 = .1 * np.random.rand(n2)

nperms = twosample.count_permutations(n1, n2)

magics = np.asarray(range(nperms))

t = twosample.stat_mfx(y1, v1, y2, v2, id='student_mfx', Magics=magics)

plt.hist(t, 101)
plt.show()
