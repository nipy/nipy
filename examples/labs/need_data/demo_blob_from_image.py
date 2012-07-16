# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This scipt generates a noisy activation image image
and extracts the blob from it.

This creaste as output
- a label image representing the nested blobs,
- an image of the averge signal per blob and
- an image with the terminal blob only

Author : Bertrand Thirion, 2009
"""
#autoindent

import os
import os.path as op

from nibabel import load, save, Nifti1Image

import nipy.labs.spatial_models.hroi as hroi
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_image

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset


# data paths
input_image = op.join(DATA_DIR, 'spmT_0029.nii.gz')
if not op.exists(input_image):
    get_second_level_dataset()
swd = os.getcwd()

# parameters
threshold = 3.0 # blob-forming threshold
smin = 5 # size threshold on bblobs

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
save(wim, op.join(swd, "blob.nii"))

# saving the image of the average-signal-per-blob
descrip = "blob average signal extracted from %s" % input_image
wim = nroi.to_image('activation', roi=True, descrip=descrip)
save(wim, op.join(swd, "bmap.nii"))

# saving the image of the end blobs or leaves
lroi = nroi.copy()
lroi.reduce_to_leaves()

descrip = "blob image extracted from %s" % input_image
wim = lroi.to_image('id', roi=True, descrip=descrip)
save(wim, op.join(swd, "leaves.nii"))

print "Wrote the blob image in %s" % op.join(swd, "blob.nii")
print "Wrote the blob-average signal image in %s" % op.join(swd, "bmap.nii")
print "Wrote the end-blob image in %s" % op.join(swd, "leaves.nii")
