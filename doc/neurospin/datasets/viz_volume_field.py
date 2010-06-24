# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Use Mayavi to visualize the structure of a VolumeData
"""

from enthought.mayavi import mlab
import numpy as np

s = np.random.random((5, 5, 5))

# Put the side at 0 

s[0, ...]  = 0
s[-1, ...] = 0
s[:, 0, :] = 0
s[:, -1, :] = 0
s[..., 0] = 0
s[..., -1] = 0

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.clf()

src = mlab.pipeline.scalar_field(s)

mlab.pipeline.volume(src, vmin=0, vmax=0.9)
# We save as a different filename than the one used, as we modify the
# curves.
mlab.savefig('volume_field_raw.jpg')
mlab.show()



