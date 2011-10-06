# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This is a little demo that simply shows ROI manipulation within
the nipy framework

Author: Bertrand Thirion, 2009-2010
"""
print __doc__

import numpy as np
import os
import matplotlib.pylab as mp
from nibabel import load, save, Nifti1Image

import nipy.labs.spatial_models.mroi as mroi
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_image
import nipy.labs.spatial_models.hroi as hroi

import get_data_light
import tempfile


# paths
data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
input_image = os.path.join(data_dir, 'spmT_0029.nii.gz')
mask_image = os.path.join(data_dir, 'mask.nii.gz')
if (not os.path.exists(input_image)) or (not os.path.exists(mask_image)):
    get_data_light.get_second_level_dataset()

# write dir
swd = tempfile.mkdtemp()

# -----------------------------------------------------
# example 1: create the ROI froma a given position
# -----------------------------------------------------

position = np.array([[0, 0, 0]])
domain = grid_domain_from_image(mask_image)
roi = mroi.subdomain_from_balls(domain, position, np.array([5.0]))

roi_domain = domain.mask(roi.label > -1)
dom_img = roi_domain.to_image()
save(dom_img, os.path.join(swd, "myroi.nii"))
print 'Wrote an ROI mask image in %s' % os.path.join(swd, "myroi.nii")
# fixme: pot roi feature ...

# ----------------------------------------------------
# ---- example 2: create ROIs from a blob image ------
# ----------------------------------------------------

# --- 2.a create the  blob image
# parameters
threshold = 3.0  # blob-forming threshold
smin = 5  # size threshold on bblobs

# prepare the data
nim = load(input_image)
affine = nim.get_affine()
shape = nim.get_shape()
data = nim.get_data()
values = data[data != 0]

# compute the  nested roi object
nroi = hroi.HROI_as_discrete_domain_blobs(domain, values,
                                          threshold=threshold, smin=smin)

# saving the blob image,i. e. a label image 

wim = nroi.to_image('id', roi=True)
descrip = "blob image extracted from %s" % input_image
blobPath = os.path.join(swd, "blob.nii")
save(wim, blobPath)

# --- 2.b take blob having id "132" as an ROI
roi = nroi.copy()
roi.select_roi([132])
wim2 = roi.to_image()
roi_path_2 = os.path.join(swd, "roi_blob_1.nii")
save(wim2, roi_path_2)

# --- 2.c take the blob closest to 'position as an ROI'
roi = mroi.subdomain_from_position_and_image(wim, position[0])
wim3 = roi.to_image()
roi_path_3 = os.path.join(swd, "blob_closest_to_%d_%d_%d.nii"
                          % (position[0][0], position[0][1], position[0][2]))
save(wim3, roi_path_3)

# --- 2.d make a set of ROIs from all the blobs
roi = mroi.subdomain_from_image(blobPath)
data = load(input_image).get_data().ravel()
feature_activ = [data[roi.select_id(id, roi=False)] for id in roi.get_id()]
roi.set_feature('activ', feature_activ)
roi.plot_feature('activ')
wim4 = roi.to_image()
roi_path_4 = os.path.join(swd, "roi_all_blobs.nii")
save(wim4, roi_path_4)

# ---- 2.e the same, a bit more complex
valid_roi = roi.get_id()[roi.representative_feature('activ') > 4.0]
roi.select_roi(valid_roi)
wim5 = roi.to_image()
roi_path_5 = os.path.join(swd, "roi_some_blobs.nii")
save(wim5, roi_path_5)

print  "Wrote ROI mask images in %s, \n %s \n %s \n and %s" %\
      (roi_path_2, roi_path_3, roi_path_4, roi_path_5)

mp.show()
