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
from nipy.utils import example_data
import get_data_light

# get the data
data_dir = get_data_light.get_it()

# First example, with a anatomical template
img     = load(os.path.join(data_dir, 'spmT_0029.nii.gz'))
data    = img.get_data()
affine  = img.get_affine()

viz.plot_map(data, affine, cut_coords=(-52, 10, 22),
                        threshold=2.0, cmap=viz.cm.cold_hot)

# Second example, with a given anatomical image
try:
    anat_img = load(example_data.get_filename('neurospin',
                                'sulcal2000', 'nobias_anubis.nii.gz'))
    viz.plot_map(data, affine, anat=anat_img.get_data(),
                anat_affine=anat_img.get_affine(),
                threshold=2, cmap=viz.cm.cold_hot)
except OSError, e:
    # File does not exist: the data package are not installed
    print e
pl.show()
