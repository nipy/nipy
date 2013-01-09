#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
Example of plotting a registration checker with nipy.labs visualization tools

The idea is to represent the anatomical image to be checked with an overlay of
the edges of the reference image. This idea is borrowed from FSL.

Needs the *templates* data package.

Needs matplotlib.
"""
print(__doc__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nipy.labs import viz
from nipy.labs.viz_tools import anat_cache

# Get the data. Here we are using the reference T1 image
anat, affine, _ = anat_cache._AnatCache.get_anat()

# Here we use the same image as a reference. As a result it is perfectly
# aligned.
reference = anat
reference_affine = affine

slicer = viz.plot_anat(anat, affine, dim=.2, black_bg=True)
slicer.edge_map(reference, reference_affine)

plt.show()
