# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of plotting a registration checker with nipy.labs vizualization tools

The idea is to represent the anatomical image to be checked with
an overlay of the edges of the reference image. This idea is borrowed
from FSL.
"""
print __doc__

import pylab as pl
from nipy.labs import viz

# Get the data. Here we are using the reference T1 image
from nipy.labs.viz_tools import anat_cache
anat, affine, _ = anat_cache._AnatCache.get_anat()

# Here we use the same image as a reference. As a result it is
# perfect aligned.
reference = anat
reference_affine = affine

slicer = viz.plot_anat(anat, affine, dim=.2, black_bg=True)
slicer.edge_map(reference, reference_affine)

pl.show()
