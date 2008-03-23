"""Simple example showing overlays in the nipy viewer.

"""

import pylab

from neuroimaging.core.image import image
from neuroimaging.data import MNI_file
from neuroimaging.data import avganat_file

import neuroimaging.ui.sliceplot as spt

# Create a SliceViewer and load the anatomical
viewer = spt.SliceViewer(MNI_file)

# load overlay img
avgimg = image.load(avganat_file)
# add it to the viewer
viewer.set_overlay(avgimg)
# change overlay cmap to Reds
viewer.olaycmap = pylab.cm.Reds
# change overlay alpha value
viewer.alpha = 0.4

pylab.show()
