# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Use Mayavi to visualize the structure of a VolumeGrid
"""

from enthought.mayavi import mlab
import numpy as np

from enthought.tvtk.api import tvtk

dims = (4, 4, 4)
x, y, z = np.mgrid[0.:dims[0], 0:dims[1], 0:dims[2]]
x = np.reshape(x.T, (-1,))
y = np.reshape(y.T, (-1,))
z = np.reshape(z.T, (-1,))
y += 0.3*np.sin(x)
z += 0.4*np.cos(x)
x += 0.05*y**3 
sgrid = tvtk.StructuredGrid(dimensions=(dims[0], dims[1], dims[2]))
sgrid.points = np.c_[x, y, z]
s = np.random.random((dims[0]*dims[1]*dims[2]))
sgrid.point_data.scalars = np.ravel(s.copy())
sgrid.point_data.scalars.name = 'scalars'

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.clf()

mlab.pipeline.surface(sgrid, opacity=0.4)
mlab.pipeline.surface(mlab.pipeline.extract_edges(sgrid), color=(0, 0, 0))
mlab.pipeline.glyph(sgrid, mode='cube', scale_factor=0.2, scale_mode='none')
mlab.savefig('volume_grid.jpg')
mlab.show()



