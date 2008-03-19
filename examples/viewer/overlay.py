"""Simple example showing overlays in the nipy viewer.

"""

import pylab

from neuroimaging.core.image import image
from neuroimaging.testing import anatfile
from neuroimaging.testing import funcfile

import neuroimaging.ui.sliceplot as spt

# Create a SliceViewer and load the anatomical
viewer = spt.SliceViewer(anatfile)

# load overlay img
avgimg = image.load(funcfile)
# add it to the viewer
viewer.set_overlay(avgimg)
# change overlay cmap to Reds
viewer.olaycmap = pylab.cm.Reds
# change overlay alpha value
viewer.alpha = 0.4

pylab.show()
