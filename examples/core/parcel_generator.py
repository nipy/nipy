#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
Example showing how to use the parcel generator.

We load an image with ROI definitions and calculate the number of voxels in each
ROI.
"""
print(__doc__)

from os.path import dirname, join as pjoin

import nipy

from nipy.core.utils.generators import parcels

OUR_PATH = dirname(__file__)
DATA_PATH = pjoin(OUR_PATH, '..', 'data')
BG_IMAGE_FNAME = pjoin(DATA_PATH, 'mni_basal_ganglia.nii.gz')

bg_img = nipy.load_image(BG_IMAGE_FNAME)
bg_data = bg_img.get_data()

"""
I happen to know that the image has these codes:

14 - Left striatum
16 - Right striatum
39 - Left caudate
53 - Right caudate

All the other voxels are zero, I don't want those.
"""

print("Number of voxels for L, R striatum; L, R caudate")
for mask in parcels(bg_data, exclude=(0,)):
    print(mask.sum())

""" Given we know the codes we can also give them directly """
print("Again with the number of voxels for L, R striatum; L, R caudate")
for mask in parcels(bg_data, labels=(14, 16, 39, 53)):
    print(mask.sum())
