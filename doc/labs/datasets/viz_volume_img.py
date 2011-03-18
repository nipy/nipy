# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Use Mayavi to visualize the structure of a VolumeImg
"""

from enthought.mayavi import mlab
import numpy as np

rand = np.random.RandomState(1)
data = rand.random_sample((5, 4, 4))

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.clf()

src = mlab.pipeline.scalar_field(data)
src.image_data.spacing = (0.5, 1, 0.7)
src.image_data.update_data()

mlab.pipeline.surface(src, opacity=0.4)
mlab.pipeline.surface(mlab.pipeline.extract_edges(src), color=(0, 0, 0))
mlab.pipeline.glyph(src, mode='cube', scale_factor=0.2, scale_mode='none')
mlab.savefig('volume_img.jpg')
mlab.show()


