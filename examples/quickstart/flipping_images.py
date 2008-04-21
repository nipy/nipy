#!/bin/env python
'''
Show use of grids / mappings by applying Y flip to image

Usage
flipping_image.py

In this example, we flip the Y coordinate of an image.
This does not change the data associated to the image, but if
we want to look at the data in a fixed coordinate system, the
data will be reflected through the Y axis.
'''

import os

import numpy as np
import numpy.linalg as L
import pylab

from neuroimaging.core.api import Image, SamplingGrid, Affine, load_image, save_image
from neuroimaging.core.reference.coordinate_system import CoordinateSystem
from neuroimaging.ui.visualization.viewer import BoxViewer
from neuroimaging.testing import anatfile, funcfile

# TODO: would be nice

subject_no=0
run_no = 1
f = load_image(funcfile)
a = f[0]
t = a.grid.mapping.transform[1:]
m = SamplingGrid.from_affine(Affine(t), ['zspace', 'yspace', 'xspace'], a.shape)
img = Image(np.asarray(a), m)

# Make Y flip transformation matrix
t = np.identity(4)
t[1,1] = -1.
flip = Affine(t)

# Make new flipped mapping and make flipped copy of image
flipped_map = flip*img.grid.mapping
anames = ['zspace', 'yspace', 'xspace']
new_grid = SamplingGrid.from_affine(flipped_map, anames, img.grid.shape)
flipped_img = Image(np.asarray(img), new_grid)
print img.affine
print flipped_img.affine

# Data is the same
print np.alltrue(np.equal(np.asarray(img), np.asarray(flipped_img)))

# Show unflipped and flipped images
#BoxViewer(mask_img).draw()
#BoxViewer(flipped_img).draw()

# Write flipped image to file
save_image(flipped_img, "flip.nii", clobber=True)

# Show that it is the grid rather than the array determining the flip
still_flipped_img = load_image('flip.nii')
print still_flipped_img.affine

from neuroimaging.core.reference.slices import bounding_box

# Show the figures -- do some resampling

from neuroimaging.core.reference.slices import xslice
import scipy.ndimage.interpolation as I
xsl = xslice(30., [-1,8.], [-60,60], img.grid.output_coords, (100,100))

tunflip = np.dot(L.inv(img.affine), xsl.affine)
print tunflip
unflip = I.affine_transform(np.asarray(img), tunflip[:3,:-1],
                            offset=tunflip[:3,-1],
                            output_shape=(100,100))

tflip = np.dot(L.inv(flipped_img.affine), xsl.affine)
flip = I.affine_transform(np.asarray(flipped_img), tflip[:3,:-1],
                          offset=tflip[:3,-1],
                          output_shape=(100,100))
os.remove("flip.nii")

# TODO -- put proper axes on these images

pylab.imshow(unflip)
pylab.title("Image of original data in X-Z plane")

pylab.figure()
pylab.imshow(flip)
pylab.title("Image of flipped data in X-Z plane")

pylab.show()
