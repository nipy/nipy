#!/usr/bin/env python3
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
__doc__ = """
Example of activation image visualization with nipy.labs visualization tools

Needs *example data* package.

Needs matplotlib
"""
print(__doc__)

import os.path

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

# Local import
from get_data_light import get_second_level_dataset
from nibabel import load

from nipy.labs import viz
from nipy.utils import example_data

# get the data
data_dir = get_second_level_dataset()

# First example, with a anatomical template
img = load(os.path.join(data_dir, 'spmT_0029.nii.gz'))
data = img.get_fdata()
affine = img.affine

viz.plot_map(data, affine, cut_coords=(-52, 10, 22),
                        threshold=2.0, cmap=viz.cm.cold_hot)
plt.savefig('ortho_view.png')

# Second example, with a given anatomical image slicing in the Z direction
try:
    anat_img = load(example_data.get_filename('neurospin', 'sulcal2000',
                                              'nobias_anubis.nii.gz'))
    anat = anat_img.get_fdata()
    anat_affine = anat_img.affine
except OSError as e:
    # File does not exist: the data package is not installed
    print(e)
    anat = None
    anat_affine = None

viz.plot_map(data, affine, anat=anat, anat_affine=anat_affine,
             slicer='z', threshold=2, cmap=viz.cm.cold_hot, black_bg=True)
plt.savefig('z_view.png')

viz.plot_map(data, affine, anat=anat, anat_affine=anat_affine,
             slicer='x', threshold=2, cmap=viz.cm.cold_hot, black_bg=True)
plt.savefig('x_view.png')

viz.plot_map(data, affine, anat=anat, anat_affine=anat_affine,
             slicer='y', threshold=2, cmap=viz.cm.cold_hot, black_bg=True)
plt.savefig('y_view.png')

plt.show()
