from neuroimaging.core.image import Image
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.ui.visualization.viewer import BoxViewer
import numpy as N
import pylab

def mask(subject=0, run=1):
    return Image('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/mask.img' % (subject,run))    

m = mask()

t = N.identity(4)
t[1,1] = -1.

input_coords = output_coords = m.grid.mapping.output_coords
flip = Affine(input_coords, output_coords, t)

newgrid = SamplingGrid(shape=m.grid.shape, mapping=flip * m.grid.mapping)
flipped = Image(m[:], grid=newgrid)

v1 = BoxViewer(m); v2 = BoxViewer(flipped)
v1.draw(); v2.draw()


flipped.tofile('flip.img')
flipped3 = Image('flip.img', grid=newgrid)
v3 = BoxViewer(flipped3); v3.draw()

pylab.show()
