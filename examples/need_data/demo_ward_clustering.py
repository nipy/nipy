#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
This shows the effect of ward clustering on a real fMRI dataset

Author: Bertrand Thirion, 2010
"""
print(__doc__)

from os import mkdir, getcwd, path

import numpy as np

from nibabel import load, save, Nifti1Image

from nipy.algorithms.graph.field import Field

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset

# paths
input_image = path.join(DATA_DIR, 'spmT_0029.nii.gz')
mask_image = path.join(DATA_DIR, 'mask.nii.gz')
if (not path.exists(mask_image)) or (not path.exists(input_image)):
    get_second_level_dataset()

# write directory
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

# read the data
mask = load(mask_image).get_data() > 0
ijk = np.array(np.where(mask)).T
nvox = ijk.shape[0]
data = load(input_image).get_data()[mask]
image_field = Field(nvox)
image_field.from_3d_grid(ijk, k=6)
image_field.set_field(data)
u, _ = image_field.ward(100)

# write the results
label_image = path.join(write_dir, 'label.nii')
wdata = mask - 1
wdata[mask] = u
save(Nifti1Image(wdata, load(mask_image).get_affine()), label_image)
print("Label image written in %s" % label_image)
