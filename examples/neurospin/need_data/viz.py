# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of activation image vizualization
with nip.neurospin vizualization tools
"""

print __doc__

import os.path as op
import pylab as pl
from nipy.io.imageformats import load
from nipy.neurospin.viz import plot_map
import get_data_light

# get the data
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data'))
data_path = op.join(data_dir,'spmT_0029.nii.gz')
if op.exists(data_path)==False:
    get_data_light.getIt()
fim = load(data_path)
fmap = fim.get_data()
affine = fim.get_affine()

#vizualization parameters
x, y, z = -52, 10, 22
threshold = 2.0
kwargs={'cmap':pl.cm.hot,'alpha':0.7,'vmin':threshold,'anat':None}


plot_map(fmap, affine, cut_coords=(x, y, z), **kwargs)
pl.show()
