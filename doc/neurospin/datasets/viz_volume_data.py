# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Use Mayavi to visualize the structure of a VolumeData
"""

from enthought.mayavi import mlab
import numpy as np

x, y, z, s = np.random.random((4, 20))

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.clf()

src = mlab.pipeline.scalar_scatter(x, y, z, s)
sgrid = mlab.pipeline.delaunay3d(src)

mlab.pipeline.surface(sgrid, opacity=0.4)
mlab.pipeline.surface(mlab.pipeline.extract_edges(sgrid), color=(0, 0, 0))
mlab.pipeline.glyph(sgrid, mode='cube', scale_factor=0.05, scale_mode='none')
mlab.savefig('volume_data.jpg')
mlab.show()



