# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of dimension reduction on a swiss roll using Isomap.
Bertrand Thirion, 2008-2010
"""
print __doc__

import numpy as np
from nipy.neurospin.eda.dimension_reduction import swiss_roll, isomap, CCA,\
     mds, MDS, knn_Isomap , PCA
import nipy.neurospin.graph.graph as fg
import matplotlib.pylab as mp

# Generate a swiss roll datasets
# y are the 3D coordinates
# x are  he  2D latent coordinates
nbsamp = 1000
y, x = swiss_roll(nbsamp)

M = knn_Isomap(y, rdim=3)
u = M.train(k=7)

# cheack that u and x span the same space,
# i.e. their two canonical coorelations are close to 1
sv = CCA(x-x.mean(0),u[:,:2])
print 'the canonical correlations between true parameters and estimated are %f, %f' %(sv[0],sv[1])

ax = M.G.show(u[:,:2])
ax.set_title('embedding of the data graph')
mp.show()


