import numpy as np

from nipy.neurospin import Image
from nipy.neurospin.statistical_mapping import cluster_stats

dx = 5
dy = 5
dz = 4

zimg = Image(3.*(np.random.rand(dx,dy,dz)-.5))
mask = Image(np.random.randint(2, size=[dx,dy,dz]))

nulls = {'smax': 10*np.random.rand(1000)}

clusters, info = cluster_stats(zimg, mask, 0.5, nulls=nulls)


