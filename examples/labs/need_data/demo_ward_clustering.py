# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This shows the effect of ward clustering on a simulated fMRI dataset

Author: Bertrand Thirion, 2010
"""
print __doc__

import os

import numpy as np

from nibabel import load, save, Nifti1Image

from nipy.algorithms.graph.field import Field

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset

# paths
swd = os.getcwd()
input_image = os.path.join(DATA_DIR, 'spmT_0029.nii.gz')
mask_image = os.path.join(DATA_DIR, 'mask.nii.gz')

if (not os.path.exists(mask_image)) or (not os.path.exists(input_image)):
    get_second_level_dataset()

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
label_image = os.path.join(swd, 'label.nii')
wdata = mask - 1
wdata[mask] = u
save(Nifti1Image(wdata, load(mask_image).get_affine()), label_image)
print "Label image written in %s" % label_image
