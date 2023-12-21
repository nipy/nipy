#!/usr/bin/env python3
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
__doc__ = """
This shows the effect of ward clustering on a real fMRI dataset

Author: Bertrand Thirion, 2010
"""
print(__doc__)

from os import getcwd, mkdir, path

import numpy as np

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset
from nibabel import Nifti1Image, load, save

from nipy.algorithms.graph.field import Field

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
mask = load(mask_image).get_fdata() > 0
ijk = np.array(np.where(mask)).T
nvox = ijk.shape[0]
data = load(input_image).get_fdata()[mask]
image_field = Field(nvox)
image_field.from_3d_grid(ijk, k=6)
image_field.set_field(data)
u, _ = image_field.ward(100)

# write the results
label_image = path.join(write_dir, 'label.nii')
wdata = mask - 1
wdata[mask] = u
save(Nifti1Image(wdata, load(mask_image).affine), label_image)
print(f"Label image written in {label_image}")
