import numpy as N
import pylab

from neuroimaging.core.image.image import Image
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.ui.visualization.viewer import BoxViewer
from neuroimaging.utils.tests.data import repository


def mask(subject=0, run=1):
    return Image('FIAC/fiac%d/fonc%d/fsl/mask.img' % (subject,run), repository) 

m = mask()

t = N.identity(4)
t[1,1] = -1.


flip = Affine(t)


flipped_map = flip*m.grid.mapping
new_grid = SamplingGrid.from_affine(flipped_map, m.grid.shape)

flipped = Image(m[:], grid=new_grid)


v1 = BoxViewer(m); v2 = BoxViewer(flipped)
v1.draw(); v2.draw()

flipped.tofile('flip.img', clobber=True)
flipped3 = Image('flip.img', grid=new_grid)
v3 = BoxViewer(flipped3); v3.draw()

pylab.show()
