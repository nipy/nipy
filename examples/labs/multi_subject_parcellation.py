#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
This script contains a quick demo on  a multi-subject parcellation on a toy 2D
example.

Note how the middle parcels adapt to the individual configuration.

Needs matplotlib
"""
print(__doc__)

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

import nipy.labs.spatial_models.hierarchical_parcellation as hp
import nipy.labs.utils.simul_multisubject_fmri_dataset as simul
import nipy.labs.spatial_models.discrete_domain as dom

# step 1:  generate some synthetic data
n_subj = 10
shape = (60, 60)
pos = 3 * np.array([[6, 7],
                  [10, 10],
                  [15, 10]])
ampli = np.array([5, 7, 6])
sjitter = 6.0
dataset = simul.surrogate_2d_dataset(n_subj=n_subj, shape=shape, pos=pos,
                                     ampli=ampli, width=10.0)
# dataset represents 2D activation images from n_subj subjects,

# step 2 : prepare all the information for the parcellation
nbparcel = 10
ldata = np.reshape(dataset, (n_subj, np.prod(shape), 1))
domain = dom.grid_domain_from_shape(shape)

# step 3 : run the algorithm
Pa = hp.hparcel(domain, ldata, nbparcel, mu=3.0)
# note: play with mu to change the 'stiffness of the parcellation'

# step 4:  look at the results
Label = np.array([np.reshape(Pa.individual_labels[:, s], shape)
                  for s in range(n_subj)])

plt.figure(figsize=(8, 4))
plt.title('Input data')
for s in range(n_subj):
    plt.subplot(2, 5, s + 1)
    plt.imshow(dataset[s], interpolation='nearest')
    plt.axis('off')

plt.figure(figsize=(8, 4))
plt.title('Resulting parcels')
for s in range(n_subj):
    plt.subplot(2, 5, s+1)
    plt.imshow(Label[s], interpolation='nearest', vmin=-1, vmax=nbparcel)
    plt.axis('off')
plt.show()
