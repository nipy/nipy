#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
Simple demo that partitions a smooth field into 10 clusters.  In most cases,
Ward's clustering behaves best.

Requires matplotlib

Author: Bertrand Thirion, 2009
"""
print(__doc__)

import numpy as np
import numpy.random as nr

from scipy.ndimage import gaussian_filter

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nipy.algorithms.graph.field import Field

dx = 50
dy = 50
dz = 1
nbseeds = 10
data = gaussian_filter( np.random.randn(dx, dy), 2)
F = Field(dx * dy * dz)
xyz = np.reshape(np.indices((dx, dy, dz)), (3, dx * dy * dz)).T.astype(np.int)
F.from_3d_grid(xyz, 6)
F.set_field(data)

seeds = np.argsort(nr.rand(F.V))[:nbseeds]
seeds, label, J0 = F.geodesic_kmeans(seeds)
wlabel, J1 = F.ward(nbseeds)
seeds, label, J2 = F.geodesic_kmeans(seeds, label=wlabel.copy(), eps=1.e-7)

print('Inertia values for the 3 algorithms: ')
print('Geodesic k-means: ', J0, 'Wards: ', J1, 'Wards + gkm: ', J2)

plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.reshape(data, (dx, dy)), interpolation='nearest')
plt.title('Input data')
plt.subplot(1, 3, 2)
plt.imshow(np.reshape(wlabel, (dx, dy)), interpolation='nearest')
plt.title('Ward clustering \n into 10 components')
plt.subplot(1, 3, 3)
plt.imshow(np.reshape(label, (dx, dy)), interpolation='nearest')
plt.title('geodesic kmeans clust. \n into 10 components')
plt.show()
