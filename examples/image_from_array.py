# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Create a nifti image from a numpy array and an affine transform.
"""

import os

import numpy as np

from nipy.core.api import Image, AffineTransform
from nipy.io.api import save_image, load_image

# This is just the filename for a tiny example file
from nipy.testing import anatfile

# Load an image to get the array and affine
#
# Use one of our test files to get an array and affine (as numpy array) from.
img = load_image(anatfile)
arr = np.asarray(img)
affine_array = img.coordmap.affine.copy()

################################################################################
# START HERE
################################################################################

# 1) Create a CoordinateMap from the affine transform which specifies
# the mapping from input to output coordinates.

# Specify the axis order of the input coordinates
input_coords = ['k', 'j', 'i']
output_coords = ['z','y','x']
#or
innames = ('kji')
outnames = ('zyx')
# either way works

# Build a CoordinateMap to create the image with
affine_coordmap = AffineTransform(innames, outnames, affine_array)

# 2) Create a nipy image from the array and CoordinateMap

# Create new image
newimg = Image(arr, affine_coordmap)

################################################################################
# END HERE, for testing purposes only.
################################################################################
# Imports used just for development and testing.  Users typically
# would not use these when creating an image.
from tempfile import mkstemp
from nipy.testing import assert_equal

# We use a temporary file for this example so as to not create junk
# files in the nipy directory.
fd, name = mkstemp(suffix='.nii.gz')
tmpfile = open(name)

# Save the nipy image to the specified filename
save_image(newimg, tmpfile.name)

# Reload and verify the affine was saved correctly.
tmpimg = load_image(tmpfile.name)
assert_equal(np.mean(tmpimg), np.mean(img))
np.testing.assert_almost_equal(np.std(tmpimg), np.std(img))
# np.testing.assert_almost_equal(np.asarray(tmpimg), np.asarray(img), -1)
# assert_equal(img.affine, tmpimg.affine)

# cleanup our tempfile
tmpfile.close()
os.unlink(name)
