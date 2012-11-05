#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Create a nifti image from a numpy array and an affine transform.
"""

import numpy as np

from nipy import save_image, load_image
from nipy.core.api import Image, vox2scanner

# This gets the filename for a tiny example file
from nipy.testing import anatfile

# Load an image to get an array and affine
#
# Use one of our test files to get an array and affine (as numpy array) from.
img = load_image(anatfile)
arr = img.get_data()
affine_array = img.coordmap.affine.copy()

# 1) Create a CoordinateMap from the affine transform which specifies
# the mapping from input to output coordinates. The ``vox2scanner`` function
# makes a coordinate map from voxels to scanner coordinates.  Other options are
# ``vox2mni`` or ``vox2talairach``
affine_coordmap = vox2scanner(affine_array)

# 2) Create a nipy image from the array and CoordinateMap
newimg = Image(arr, affine_coordmap)

# Save the nipy image to the specified filename
save_image(newimg, 'an_image.nii.gz')

# Reload and verify the data and affine were saved correctly.
img_back = load_image('an_image.nii.gz')
assert np.allclose(img_back.get_data(), img.get_data())
assert np.allclose(img_back.coordmap.affine, img.coordmap.affine)
