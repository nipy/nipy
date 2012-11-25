#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
"""
This script generates a noisy activation image and extracts the blob
from it.

This creates as output
- a label image representing the nested blobs,
- an image of the average signal per blob and
- an image with the terminal blob only

Author : Bertrand Thirion, 2009
"""
#autoindent

from os import mkdir, getcwd, path

from nibabel import load, save, Nifti1Image

import nipy.labs.spatial_models.hroi as hroi
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_image

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset


# data paths
input_image = path.join(DATA_DIR, 'spmT_0029.nii.gz')
if not path.exists(input_image):
    get_second_level_dataset()
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

# parameters
threshold = 3.0 # blob-forming threshold
smin = 5 # size threshold on blobs

# prepare the data
nim = load(input_image)
mask_image = Nifti1Image((nim.get_data() ** 2 > 0).astype('u8'),
                         nim.get_affine())
domain = grid_domain_from_image(mask_image)
data = nim.get_data()
values = data[data != 0]

# compute the  nested roi object
nroi = hroi.HROI_as_discrete_domain_blobs(domain, values, threshold=threshold,
                                          smin=smin)

# compute region-level activation averages
activation = [values[nroi.select_id(id, roi=False)] for id in nroi.get_id()]
nroi.set_feature('activation', activation)
average_activation = nroi.representative_feature('activation')

# saving the blob image,i. e. a label image
descrip = "blob image extracted from %s" % input_image
wim = nroi.to_image('id', roi=True, descrip=descrip)
save(wim, path.join(write_dir, "blob.nii"))

# saving the image of the average-signal-per-blob
descrip = "blob average signal extracted from %s" % input_image
wim = nroi.to_image('activation', roi=True, descrip=descrip)
save(wim, path.join(write_dir, "bmap.nii"))

# saving the image of the end blobs or leaves
lroi = nroi.copy()
lroi.reduce_to_leaves()

descrip = "blob image extracted from %s" % input_image
wim = lroi.to_image('id', roi=True, descrip=descrip)
save(wim, path.join(write_dir, "leaves.nii"))

print("Wrote the blob image in %s" % path.join(write_dir, "blob.nii"))
print("Wrote the blob-average signal image in %s"
      % path.join(write_dir, "bmap.nii"))
print("Wrote the end-blob image in %s" % path.join(write_dir, "leaves.nii"))
