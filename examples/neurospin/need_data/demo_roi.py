# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This is a little demo that simply shows ROI manipulation within
the nipy framework

Author: Bertrand Thirion, 2009
"""
print __doc__

import numpy as np
import os
import matplotlib.pylab as mp
from nipy.io.imageformats import load, save, Nifti1Image 

from nipy.neurospin.spatial_models.roi import DiscreteROI, MultipleROI
from nipy.neurospin.spatial_models.mroi import SubDomains, subdomain_from_balls
from nipy.neurospin.spatial_models.discrete_domain import domain_from_image, grid_domain_from_image
import nipy.neurospin.spatial_models.hroi as hroi

import get_data_light
import tempfile
#get_data_light.get_it()

# paths
swd = tempfile.mkdtemp()
data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
input_image = os.path.join(data_dir,'spmT_0029.nii.gz')
mask_image = os.path.join(data_dir,'mask.nii.gz')

# -----------------------------------------------------
# example 1: create the ROI froma a given position
# -----------------------------------------------------

position = np.array([[0, 0, 0]])
domain = grid_domain_from_image(mask_image)
roi = subdomain_from_balls(domain, position, np.array([5.0]))
roi_domain = domain.mask(roi.label>-1)
roi_domain.to_image(os.path.join(swd, "myroi.nii"))
print 'Wrote an ROI mask image in %s' %os.path.join(swd, "myroi.nii")
# fixme: pot roi feature ...

# ----------------------------------------------------
# ---- example 2: create ROIs from a blob image ------
# ----------------------------------------------------

# --- 2.a create the  blob image
# parameters
threshold = 3.0 # blob-forming threshold
smin = 5 # size threshold on bblobs

# prepare the data
nim = load(input_image)
affine = nim.get_affine()
shape = nim.get_shape()
data = nim.get_data()
values = data[data!=0]

# compute the  nested roi object
nroi = hroi.HROI_as_discrete_domain_blobs(domain, values,
                                          threshold=threshold, smin=smin)

# compute region-level activation averages
nroi.make_feature('activation', values)
average_activation = nroi.representative_feature('activation')
bmap = -np.zeros(domain.size)
for k in range(nroi.k):
    bmap[nroi.label==k] = average_activation[k]

# saving the blob image,i. e. a label image 
wlabel = -2*np.ones(shape)
wlabel[data!=0] = nroi.label
blobPath = os.path.join(swd, "blob.nii")

wim = Nifti1Image(wlabel, affine)
wim.get_header()['descrip'] = 'blob image extracted from %s'%input_image
save(wim, blobPath)

# --- 2.b take blob labelled "1" as an ROI
roi = DiscreteROI( affine=affine, shape=shape)
roi.from_labelled_image(blobPath, 1)
roiPath2 = os.path.join(swd, "roi_blob_1.nii")
roi.make_image(roiPath2)

# --- 2.c take the blob closest to 'position as an ROI'
roiPath3 = os.path.join(swd, "blob_closest_to_%d_%d_%d.nii")%\
           (position[0][0], position[0][1], position[0][2])
roi.from_position_and_image(blobPath, np.array(position))
roi.make_image(roiPath3)

# --- 2.d make a set of ROIs from all the blobs
mroi = MultipleROI( affine=affine, shape=shape)
mroi.from_labelled_image(blobPath)
roiPath4 = os.path.join(swd, "roi_all_blobs.nii")
mroi.make_image(roiPath4)
mroi.set_discrete_feature_from_image('activ', input_image)
mroi.discrete_to_roi_features('activ')
mroi.plot_roi_feature('activ')

# ---- 2.e the same, a bit more complex
mroi = MultipleROI( affine=affine, shape=shape)
mroi.as_multiple_balls(np.array([[-10.,0.,10.]]),np.array([7.0]))
mroi.from_labelled_image(blobPath,np.arange(1,20))
mroi.from_labelled_image(blobPath,np.arange(31,50))
roiPath5 = os.path.join(swd,"roi_some_blobs.nii")
mroi.set_discrete_feature_from_image('activ',input_image)
mroi.discrete_to_roi_features('activ')
valid = mroi.get_roi_feature('activ')>4.0
mroi.clean(valid)
mroi.make_image(roiPath5)

print  "Wrote ROI mask images in %s, \n %s \n %s \n and %s" %\
      (roiPath2, roiPath3, roiPath4, roiPath5)

mp.show()
