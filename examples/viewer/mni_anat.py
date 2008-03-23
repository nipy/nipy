"""Simple example showing how to load an MNI file into the nipy viewer.

I usually run this interactively from ipython and I've modified my pylab
interface so all pylab commands are in a pylab namespace.

The viewer accepts filenames, nipy image objects and numpy arrays as data:
In [6]: imgarray = img._data.copy()
In [7]: type(imgarray)
Out[7]: <class 'numpy.core.memmap.memmap'>
In [8]: viewer.set_data(imgarray)

"""

import pylab

from neuroimaging.core.image import image
from neuroimaging.data import MNI_file

import neuroimaging.ui.sliceplot as spt

# Create a SliceViewer
viewer = spt.SliceViewer()
# Load our image
img = image.load(MNI_file)
# Set the data for plots in the viewer
viewer.set_data(img)

pylab.show()
