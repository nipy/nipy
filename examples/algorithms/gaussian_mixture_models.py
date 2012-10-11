#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
Example of a demo that fits a Gaussian Mixture Model (GMM) to  a dataset The
possible number of clusters is in the [1,10] range The proposed algorithm
correctly selects a solution with 2 or 3 classes

Requires matplotlib

Author : Bertrand Thirion, 2008-2009
"""
print(__doc__)

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

import nipy.algorithms.clustering.gmm as gmm

dim = 2
# 1. generate a 3-components mixture
x1 = np.random.randn(100, dim)
x2 = 3 + 2 * np.random.randn(50, dim)
x3 = np.repeat(np.array([- 2, 2], ndmin=2), 30, 0) \
     + 0.5 * np.random.randn(30, dim)
x = np.concatenate((x1, x2, x3))

# 2. fit the mixture with a bunch of possible models
krange = range(1, 5)
lgmm = gmm.best_fitting_GMM(x, krange, prec_type='diag', niter=100,
                            delta=1.e-4, ninit=1, verbose=0)

# 3, plot the result
z = lgmm.map_label(x)
gmm.plot2D(x, lgmm, z, verbose=0)
plt.show()
