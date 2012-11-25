#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
""" Example of onesample permutation test

Needs *example data* package
"""

import numpy as np

from nipy.labs.group.permutation_test import permutation_test_onesample

from nipy.utils import example_data

# Get group data
group_data = example_data.get_filename('neurospin', 'language_babies',
                                       'offset_002.npz')

f = np.load(group_data)
data, vardata, xyz = f['mat'], f['var'], f['xyz']

# Create one-sample permutation test instance
ptest = permutation_test_onesample(data, xyz, stat_id='wilcoxon')

# Cluster definition: (threshold, diameter)
# Note that a list of definitions can be passed to ptest.calibrate
cluster_def = (ptest.height_threshold(0.01), None)
print(cluster_def)

# Multiple calibration
# To get accurate pvalues, don't pass nperms (default is 1e4)
# Yet it will take longer to run
voxel_res, cluster_res, region_res = ptest.calibrate(nperms=100,
                                                     clusters=[cluster_def])

# Simulated Zmax values for FWER correction
simu_zmax = ptest.zscore(voxel_res['perm_maxT_values'])

# Output regions
## This is a list because several cluster definitions can be accepted
clusters = cluster_res[0]
sizes = clusters['size_values']
clusters_Pcorr = clusters['size_Corr_p_values']

# Simulated cluster sizes
simu_s = clusters['perm_size_values']
simu_smax = clusters['perm_maxsize_values']
