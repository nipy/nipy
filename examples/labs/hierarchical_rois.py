# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
__doc__ = \
"""
Example of a script that crates a 'hierarchical roi' structure from the blob
model of an image

Needs matplotlib

Author: Bertrand Thirion, 2008-2009
"""
print __doc__

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

import nipy.labs.spatial_models.hroi as hroi
import nipy.labs.utils.simul_multisubject_fmri_dataset as simul
from nipy.labs.spatial_models.discrete_domain import domain_from_binary_array

##############################################################################
# simulate the data
shape = (60, 60)
pos = np.array([[12, 14], [20, 20], [30, 20]])
ampli = np.array([3, 4, 4])

dataset = simul.surrogate_2d_dataset(n_subj=1, shape=shape, pos=pos,
                                     ampli=ampli, width=10.0).squeeze()

# create a domain descriptor associated with this
domain = domain_from_binary_array(dataset ** 2 > 0)

nroi = hroi.HROI_as_discrete_domain_blobs(domain, dataset.ravel(),
                                          threshold=2.0, smin=3)

n1 = nroi.copy()
n2 = nroi.reduce_to_leaves()

td = n1.make_forest().depth_from_leaves()
root = np.argmax(td)
lv = n1.make_forest().get_descendents(root)
u = nroi.make_graph().cc()

flat_data = dataset.ravel()
activation = [flat_data[nroi.select_id(id, roi=False)]
              for id in nroi.get_id()]
nroi.set_feature('activation', activation)

label = np.reshape(n1.feature_to_voxel_map('id', roi=True), shape)

# make a figure
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(dataset))
plt.title('Input map')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Nested Rois')
plt.imshow(label, interpolation='Nearest')
plt.axis('off')
plt.show()
