#!/usr/bin/env python3
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
__doc__ = """
This is a little demo that simply shows ROI manipulation within the nipy
framework.

Needs matplotlib

Author: Bertrand Thirion, 2009-2010
"""
print(__doc__)

from os import getcwd, mkdir, path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset
from nibabel import load, save

from nipy.labs.spatial_models import hroi, mroi
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_image

# paths
input_image = path.join(DATA_DIR, 'spmT_0029.nii.gz')
mask_image = path.join(DATA_DIR, 'mask.nii.gz')
if (not path.exists(input_image)) or (not path.exists(mask_image)):
    get_second_level_dataset()

# write directory
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)


# -----------------------------------------------------
# example 1: create the ROI from a given position
# -----------------------------------------------------

position = np.array([[0, 0, 0]])
domain = grid_domain_from_image(mask_image)
roi = mroi.subdomain_from_balls(domain, position, np.array([5.0]))

roi_domain = domain.mask(roi.label > -1)
dom_img = roi_domain.to_image()
save(dom_img, path.join(write_dir, "myroi.nii"))
print(f"Wrote an ROI mask image in {path.join(write_dir, 'myroi.nii')}")

# ----------------------------------------------------
# ---- example 2: create ROIs from a blob image ------
# ----------------------------------------------------

# --- 2.a create the  blob image
# parameters
threshold = 3.0  # blob-forming threshold
smin = 10  # size threshold on bblobs

# prepare the data
nim = load(input_image)
affine = nim.affine
shape = nim.shape
data = nim.get_fdata()
values = data[data != 0]

# compute the nested roi object
nroi = hroi.HROI_as_discrete_domain_blobs(domain, values,
                                          threshold=threshold, smin=smin)

# saving the blob image, i.e. a label image
wim = nroi.to_image('id', roi=True)
descrip = f"blob image extracted from {input_image}"
blobPath = path.join(write_dir, "blob.nii")
save(wim, blobPath)

# --- 2.b take blob having id "132" as an ROI
roi = nroi.copy()
roi.select_roi([132])
wim2 = roi.to_image()
roi_path_2 = path.join(write_dir, "roi_blob_1.nii")
save(wim2, roi_path_2)

# --- 2.c take the blob closest to 'position as an ROI'
roi = mroi.subdomain_from_position_and_image(wim, position[0])
wim3 = roi.to_image()
roi_path_3 = path.join(write_dir, "blob_closest_to_%d_%d_%d.nii"
                          % (position[0][0], position[0][1], position[0][2]))
save(wim3, roi_path_3)

# --- 2.d make a set of ROIs from all the blobs
roi = mroi.subdomain_from_image(blobPath)
data = load(input_image).get_fdata().ravel()
feature_activ = [data[roi.select_id(id, roi=False)] for id in roi.get_id()]
roi.set_feature('activ', feature_activ)
roi.plot_feature('activ')
wim4 = roi.to_image()
roi_path_4 = path.join(write_dir, "roi_all_blobs.nii")
save(wim4, roi_path_4)

# ---- 2.e the same, a bit more complex
valid_roi = roi.get_id()[roi.representative_feature('activ') > 4.0]
roi.select_roi(valid_roi)
wim5 = roi.to_image()
roi_path_5 = path.join(write_dir, "roi_some_blobs.nii")
save(wim5, roi_path_5)

print("Wrote ROI mask images in {}, \n {} \n {} \n and {}".format(roi_path_2, roi_path_3, roi_path_4, roi_path_5))

plt.show()
