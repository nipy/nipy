import neuroimaging as ni
import numpy as N
import pylab

def mask(subject=0, run=1):
    return ni.image.Image('http://kff.stanford.edu/FIAC/fiac%d/fonc%d/fsl/mask.img' % (subject,run))    

m = mask()

t = N.identity(4)
t[1,1] = -1.

incoords = outcoords = m.grid.mapping.output_coords
flip = ni.reference.mapping.Affine(incoords, outcoords, t)

newgrid = ni.reference.grid.SamplingGrid(shape=m.grid.shape, mapping=flip * m.grid.mapping)
flipped = ni.image.Image(m.readall(), grid=newgrid)

v1 = ni.visualization.viewer.BoxViewer(m); v2 = ni.visualization.viewer.BoxViewer(flipped)
v1.draw(); v2.draw()


flipped.tofile('flip.img')
flipped3 = ni.image.Image('flip.img', grid=newgrid)
v3 = ni.visualization.viewer.BoxViewer(flipped3); v3.draw()

pylab.show()
