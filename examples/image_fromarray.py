"""Create a nifti image from a numpy array and an affine transform."""

import os

import numpy as np

from neuroimaging.core.api import fromarray, Affine
from neuroimaging.io.api import save_image, load_image
from neuroimaging.utils.tests.data import repository

# Load an image to get the array and affine
filename = str(repository._fullpath('avg152T1.nii.gz'))
assert os.path.exists(filename)

# Use one of our test files to get an array and affine (as numpy array) from.
img = load_image(filename)
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
affine_coordmap = Affine.from_params(innames, outnames, affine_array)

# 2) Create a nipy image from the array and CoordinateMap

# Create new image
newimg = fromarray(arr, innames=innames, outnames=outnames, 
                                         coordmap=affine_coordmap)

################################################################################
# END HERE, for testing purposes only.
################################################################################
# Imports used just for development and testing.  User's typically
# would not uses these when creating an image.
from tempfile import mkstemp
from neuroimaging.testing import assert_equal

# We use a temporary file for this example so as to not create junk
# files in the nipy directory.
fd, name = mkstemp(suffix='.nii.gz')
tmpfile = open(name)

# Save the nipy image to the specified filename
save_image(newimg, tmpfile.name)


# Reload and verify the affine was saved correctly.
tmpimg = load_image(tmpfile.name)
assert_equal(tmpimg.affine, affine_coordmap.affine)
assert_equal(np.mean(tmpimg), np.mean(img))
assert_equal(np.std(tmpimg), np.std(img))
assert_equal(np.asarray(tmpimg), np.asarray(img))

# cleanup our tempfile
tmpfile.close()
os.unlink(name)

