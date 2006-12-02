#!/bin/env python
'''
Show use of grids / mappings by applying Y flip to image

Usage
flipping_image.py
'''

import numpy as N
import pylab

from neuroimaging.core.image.image import Image
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.ui.visualization.viewer import BoxViewer
from neuroimaging.utils.tests.data import repository

subject_no=0
run_no = 1
mask_img = Image('FIAC/fiac%d/fonc%d/fsl/mask.img' %
                 (subject_no, run_no), repository) 

# Make Y flip transformation matrix
t = N.identity(4)
t[1,1] = -1.
flip = Affine(t)

# Make new flipped mapping and make flipped copy of image
flipped_map = flip*mask_img.grid.mapping
new_grid = SamplingGrid.from_affine(flipped_map, mask_img.grid.shape)
flipped_img = Image(mask_img[:], grid=new_grid)

# Show unflipped and flipped images
BoxViewer(mask_img).draw()
BoxViewer(flipped_img).draw()

# Write flipped image to file
flipped_img.tofile('flip.img', clobber=True)

# Show that it is the grid rather than the array determining the flip
still_flipped_img = Image('flip.img', grid=new_grid)
BoxViewer(still_flipped_img).draw()

# Show the figures
pylab.show()
