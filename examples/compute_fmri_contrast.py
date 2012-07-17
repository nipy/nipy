#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

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

from nibabel import load as load_image

from nipy.labs.viz import plot_map, cm
from nipy.modalities.fmri.glm import GeneralLinearModel, data_scaling
from nipy.utils import example_data


# Optional argument - default value 1, 0, 0, 0
nargs = len(sys.argv)
if nargs not in (1, 2, 5):
    print USAGE
    exit(1)
if nargs == 1: # default no-argument case
    cvect = [1, 0, 0, 0]
else:
    if nargs == 2: # contrast as one string
        args = sys.argv[1].split(',')
    elif nargs == 5: # contrast as seqence of strings
        args = [arg.replace(',', '') for arg in sys.argv[1:]]
    if len(args) != 4:
        print USAGE
        exit(1)
    try:
        cvect = [float(arg) for arg in args]
    except ValueError:
        print USAGE
        exit(1)

# Input files
fmri_files = [example_data.get_filename('fiac', 'fiac0', run)
              for run in ['run1.nii.gz', 'run2.nii.gz']]
design_files = [example_data.get_filename('fiac', 'fiac0', run)
                for run in ['run1_design.npz', 'run2_design.npz']]
mask_file = example_data.get_filename('fiac', 'fiac0', 'mask.nii.gz')

# Get design matrix as numpy array
print('Loading design matrices...')
X = [np.load(f)['X'] for f in design_files]

# Get multi-session fMRI data
print('Loading fmri data...')
Y = [load_image(f) for f in fmri_files]

# Get mask image
print('Loading mask...')
mask = load_image(mask_file)
mask_array, affine = mask.get_data() > 0, mask.get_affine()

# GLM fitting
print('Starting fit...')
glms = []
for x, y in zip(X, Y):
    glm = GeneralLinearModel(x)
    data, mean = data_scaling(y.get_data()[mask_array].T)
    glm.fit(data, 'ar1')
    glms.append(glm)

# Compute the required contrast
print('Computing test contrast image...')
nregressors = X[0].shape[1]
## should check that all design matrices have the same
c = np.zeros(nregressors)
c[0:4] = cvect
z_vals = (glms[0].contrast(c) + glms[1].contrast(c)).z_score()

# Show Zmap image
z_map = mask_array.astype(np.float)
z_map[mask_array] = z_vals
mean_map = mask_array.astype(np.float)
mean_map[mask_array] = mean
plot_map(z_map,
         affine,
         anat=mean_map,
         anat_affine=affine,
         cmap=cm.cold_hot,
         threshold=2.5,
         black_bg=True)
plt.show()
