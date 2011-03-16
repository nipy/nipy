# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Simple demo that partitions a smooth field into 10 clusters
In most cases, Ward's clustering behaves best.

Author: Bertrand Thirion, 2009
"""
print __doc__

import numpy as np
import numpy.random as nr
import nipy.labs.graph.field as ff


dx = 50
dy = 50
dz = 1
nbseeds = 10
F = ff.Field(dx * dy * dz)
xyz = np.reshape(np.indices((dx, dy, dz)), (3, dx * dy * dz)).T.astype(np.int)
F.from_3d_grid(xyz, 18)
data = nr.randn(dx * dy * dz, 1)
F.set_weights(F.get_weights() / 18)
F.set_field(data)
F.diffusion(5)
data = F.get_field()

seeds = np.argsort(nr.rand(F.V))[:nbseeds]
seeds, label, J0 = F.geodesic_kmeans(seeds)
wlabel, J1 = F.ward(nbseeds)
seeds, label, J2 = F.geodesic_kmeans(seeds, label=wlabel.copy(), eps=1.e-7)

print 'inertia values for the 3 algorithms: '
print 'geodesic k-means: ', J0, 'wards: ', J1, 'wards + gkm: ', J2

import matplotlib.pylab as mp
mp.figure()
mp.subplot(1, 3, 1)
mp.imshow(np.reshape(data, (dx, dy)), interpolation='nearest')
mp.title('Input data')
mp.subplot(1, 3, 2)
mp.imshow(np.reshape(wlabel, (dx, dy)), interpolation='nearest')
mp.title('Ward clustering \n into 10 components')
mp.subplot(1, 3, 3)
mp.imshow(np.reshape(label, (dx, dy)), interpolation='nearest')
mp.title('geodesic kmeans clust. \n into 10 components')
mp.show()
