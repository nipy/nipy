# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of activation image vizualization
with nipy.labs vizualization tools
"""
print __doc__

import os.path
import pylab as pl
from nibabel import load
from nipy.labs import viz
import get_data_light

# get the data
data_dir = get_data_light.get_second_level_dataset()

img = load(os.path.join(data_dir, 'spmT_0029.nii.gz'))
data = img.get_data()
affine = img.get_affine()

viz.plot_map(data, affine, cut_coords=(-52, 10, 22),
                        threshold=2.0, cmap=viz.cm.cold_hot)
pl.show()
