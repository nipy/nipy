# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""This example shows how to create a temporary image to use during processing.

The array is filled with zeros.
"""

import numpy as np

from nipy import load_image, save_image
from nipy.core.api import Image, AffineTransform

# create an array of zeros, the shape of your data array
zero_array = np.zeros((91,109,91))

# create an image from our array
img = Image(zero_array, AffineTransform('ijk', 'xyz', np.eye(4)))

# save the image to a file
newimg = save_image(img, 'tempimage.nii.gz')

# Example of creating a temporary image file from an existing image with a
# matching coordinate map.
img = load_image('tempimage.nii.gz')
zeroarray = np.zeros(img.shape)
zeroimg = Image(zeroarray, img.coordmap)
newimg = save_image(zeroimg, 'another_tempimage.nii.gz')
