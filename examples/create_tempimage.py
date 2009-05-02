"""This example shows how to create a temporary image to use during processing.

The array is filled with zeros.

"""

import numpy as np

from nipy.core.api import fromarray, save_image

# create an array of zeros, the shape of your data array
zero_array = np.zeros((91,109,91))

# create an image from our array
img = fromarray(zero_array)

# save the image to a file
newimg = save_image(img, 'tempimage.nii.gz')


# Example of creating a temporary image file from an existing image
# with a matching comap.

# from nipy.core.api import load_image
# img = load_image('foo.nii.gz')
# zeroarray = np.zeros(img.comap.shape)
# zeroimg = fromarray(zeroarray, comap=img.comap)
# newimg = save_image(zeroimg, 'tempimage.nii.gz')
