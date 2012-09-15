#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility

import sys

USAGE = """
usage : python %s [1x4-contrast]
where [1x4-contrast] is optional and is something like 1,0,0,0

If you don't enter a contrast, 1,0,0,0 is the default.

An activation image is displayed.

This script requires the nipy-data package to run. It is an example of using a
general linear model in single-subject fMRI data analysis context. Two sessions
of the same subject are taken from the FIAC'05 dataset.

The script also needs matplotlib installed.

Author: Alexis Roche, Bertrand Thirion, 2009--2012.
""" % sys.argv[0]

__doc__ = USAGE

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nipy.labs.viz import plot_map, cm
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.utils import example_data


# Optional argument - default value 1, 0, 0, 0
nargs = len(sys.argv)
if nargs not in (1, 2, 5):
    print(USAGE)
    exit(1)
if nargs == 1:  # default no-argument case
    cvect = [1, 0, 0, 0]
else:
    if nargs == 2:  # contrast as one string
        args = sys.argv[1].split(',')
    elif nargs == 5:  # contrast as sequence of strings
        args = [arg.replace(',', '') for arg in sys.argv[1:]]
    if len(args) != 4:
        print(USAGE)
        exit(1)
    try:
        cvect = [float(arg) for arg in args]
    except ValueError:
        print(USAGE)
        exit(1)

# Input files
fmri_files = [example_data.get_filename('fiac', 'fiac0', run)
              for run in ['run1.nii.gz', 'run2.nii.gz']]
design_files = [example_data.get_filename('fiac', 'fiac0', run)
                for run in ['run1_design.npz', 'run2_design.npz']]
mask_file = example_data.get_filename('fiac', 'fiac0', 'mask.nii.gz')

# Load all the data
multi_session_model = FMRILinearModel(fmri_files, design_files, mask_file)

# GLM fitting
multi_session_model.fit(do_scaling=True, model='ar1')

# Compute the required contrast
print('Computing test contrast image...')
n_regressors = [np.load(f)['X'].shape[1] for f in design_files]
con = [np.hstack((cvect, np.zeros(nr - len(cvect)))) for nr in n_regressors]
z_map, = multi_session_model.contrast(con)

# Show Z-map image
mean_map = multi_session_model.means[0]
plot_map(z_map.get_data(),
         z_map.get_affine(),
         anat=mean_map.get_data(),
         anat_affine=mean_map.get_affine(),
         cmap=cm.cold_hot,
         threshold=2.5,
         black_bg=True)
plt.show()
