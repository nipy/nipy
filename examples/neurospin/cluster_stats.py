import nipy.neurospin as fff2

import nipy.neurospin.neuro
from nipy.neurospin.neuro.statistical_test import cluster_stats
import numpy as np

dx = 5
dy = 5
dz = 4

zimg = fff2.neuro.image(3.*(np.random.rand(dx,dy,dz)-.5))
mask = fff2.neuro.image(np.random.randint(2, size=[dx,dy,dz]))

null_smax = 10*np.random.rand(1000)


clusters, info = cluster_stats(zimg, mask, 0.5, null_smax=null_smax)


