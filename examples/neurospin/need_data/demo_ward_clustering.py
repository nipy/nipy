# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This is a little demo that simply shows the effect of ward clustering
on an fMRI dataset

Author: Bertrand Thirion, 2010
"""
print __doc__

import numpy as np
import os
from nipy.io.imageformats import load, save, Nifti1Image 
from nipy.neurospin.graph.field import Field
import get_data_light
import tempfile
data_dir = get_data_light.get_it()

# paths
swd = tempfile.mkdtemp()
input_image = os.path.join(data_dir, 'spmT_0029.nii.gz')
mask_image = os.path.join(data_dir, 'mask.nii.gz')

mask = load(mask_image).get_data()>0
ijk = np.array(np.where(mask)).T
nvox = ijk.shape[0]
data = load(input_image).get_data()[mask]
image_field = Field(nvox)
image_field.from_3d_grid(ijk, k=6)
image_field.set_field(data)
u = image_field.ward(100)

label_image = os.path.join(swd, 'label.nii')
wdata = mask - 1
wdata[mask] = u
save(Nifti1Image(wdata, load(mask_image).get_affine()), label_image)
print "Label image written in %s"  % label_image
