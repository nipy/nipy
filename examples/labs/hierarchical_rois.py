# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
__doc__ = \
"""
Example of a script that crates a 'hierarchical roi' structure
from the blob model of an image

Author: Bertrand Thirion, 2008-2009
"""
print __doc__


import numpy as np

import nipy.labs.spatial_models.hroi as hroi
import nipy.labs.utils.simul_multisubject_fmri_dataset as simul
from nipy.labs.spatial_models.discrete_domain import domain_from_array

##############################################################################
# simulate the data
dimx = 60
dimy = 60
pos = np.array([[12, 14], [20, 20], [30, 20]])
ampli = np.array([3, 4, 4])

dataset = simul.surrogate_2d_dataset(nbsubj=1, dimx=dimx, dimy=dimy, pos=pos,
                                     ampli=ampli, width=10.0).squeeze()

# create a domain descriptor associated with this
domain = domain_from_array(dataset ** 2 > 0)

nroi = hroi.HROI_as_discrete_domain_blobs(domain, dataset.ravel(),
                                          threshold=2.0, smin=3)

n1 = nroi.copy()
n2 = nroi.reduce_to_leaves()

td = n1.make_forest().depth_from_leaves()
root = np.argmax(td)
lv = n1.make_forest().rooted_subtree(root)
u = nroi.make_graph().cc()

nroi.make_feature('activation', dataset.ravel())
nroi.representative_feature('activation')

label = np.reshape(n1.label, (dimx, dimy))

# make a figure
import matplotlib.pylab as mp
mp.figure()
mp.subplot(1, 2, 1)
mp.imshow(np.squeeze(dataset))
mp.title('Input map')
mp.axis('off')
mp.subplot(1, 2, 2)
mp.title('Nested Rois')
mp.imshow(label, interpolation='Nearest')
mp.axis('off')
mp.show()
