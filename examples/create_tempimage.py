# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""This example shows how to create a temporary image to use during processing.

The array is filled with zeros.
"""

import numpy as np

from nipy import load_image, save_image
from nipy.core.api import Image, vox2mni

# create an array of zeros, the shape of your data array
zero_array = np.zeros((91,109,91))

# create an image from our array.  The image will be in MNI space
img = Image(zero_array, vox2mni(np.diag([2, 2, 2, 1])))

# save the image to a file
newimg = save_image(img, 'tempimage.nii.gz')

# Example of creating a temporary image file from an existing image with a
# matching coordinate map.
img = load_image('tempimage.nii.gz')
zeroarray = np.zeros(img.shape)
zeroimg = Image(zeroarray, img.coordmap)
newimg = save_image(zeroimg, 'another_tempimage.nii.gz')
