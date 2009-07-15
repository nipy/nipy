"""
This is a little demo that simply shows ROI manipulation within
the nipy framework

Author: Bertrand Thirion, 2009
"""
import numpy as np
import os
import matplotlib.pylab as mp

import nifti
from nipy.neurospin.utils.roi import ROI, MultipleROI
import get_data_light
import tempfile
get_data_light.getIt()

# paths
swd = tempfile.mkdtemp()
data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
InputImage = os.path.join(data_dir,'spmT_0029.nii.gz')
MaskImage = os.path.join(data_dir,'mask.nii.gz')

# -----------------------------------------------------
# example 1: create the ROI froma a given position
# -----------------------------------------------------

position = [0,0,0]
nim = nifti.NiftiImage(MaskImage)
header = nim.header
roi = ROI("myroi",header)
roi.from_position(np.array(position),5.0)
roi.make_image(os.path.join(swd,"myroi.nii"))
roi.set_feature_from_image('activ',InputImage)
roi.plot_feature('activ')

print 'Wrote an ROI mask image in %s' %os.path.join(swd,"myroi.nii")


# ----------------------------------------------------
# ---- example 2: create ROIs from a blob image ------
# ----------------------------------------------------

# --- 2.a create the  blob image
import nipy.neurospin.graph.field as ff
import nipy.neurospin.spatial_models.hroi as hroi

# parameters
threshold = 3.0 # blob-forming threshold
smin = 5 # size threshold on bblobs

# prepare the data
nim = nifti.NiftiImage(InputImage)
header = nim.header
data = nim.asarray().T
xyz = np.array(np.where(data)).T
F = ff.Field(xyz.shape[0])
F.from_3d_grid(xyz)
F.set_field(data[data!=0])

# compute the  nested roi object
label = -np.ones(F.V)
nroi = hroi.NROI_from_field(F,header,xyz,0,threshold,smin)
bmap = -np.zeros(F.V)
if nroi!=None:
    idx = nroi.discrete_features['masked_index']
    for k in range(nroi.k):
        label[idx[k]] = k

# saving the blob image,i. e. a label image 
wlabel = -2*np.ones(nim.getVolumeExtent())
wlabel[data!=0] = label
wim =  nifti.NiftiImage(wlabel.T,nim.header)
wim.description = 'blob image extracted from %s'%InputImage 
blobPath = os.path.join(swd,"blob.nii") 
wim.save(blobPath)

# --- 2.b take blob labelled "1" as an ROI
roi = ROI(header=header)
roi.from_labelled_image(blobPath,1)
roiPath2 = os.path.join(swd,"roi_blob_1.nii")
roi.make_image(roiPath2)

# --- 2.c take the blob closest to 'position as an ROI'
roiPath3 = os.path.join(swd,"blob_closest_to_%d_%d_%d.nii")%\
           (position[0],position[1],position[2])
roi.from_position_and_image("/tmp/blob.nii",np.array(position))
roi.make_image(roiPath3)

# --- 2.d make a set of ROIs from all the blobs
mroi = MultipleROI(header=header)
mroi.from_labelled_image(blobPath)
roiPath4 = os.path.join(swd,"roi_all_blobs.nii")
mroi.make_image(roiPath4)
mroi.set_roi_feature_from_image('activ',InputImage)
mroi.plot_roi_feature('activ')

# ---- 2.e the same, a bit more complex
mroi = MultipleROI(header=header)
mroi.as_multiple_balls(np.array([[-10.,0.,10.]]),np.array([7.0]))
mroi.from_labelled_image(blobPath,np.arange(1,20))
mroi.from_labelled_image(blobPath,np.arange(31,50))
roiPath5 = os.path.join(swd,"roi_some_blobs.nii")
mroi.set_roi_feature_from_image('activ',InputImage)
valid = mroi.get_roi_feature('activ')>4.0
mroi.clean(valid)
mroi.make_image(roiPath5)

print  "Wrote ROI mask images in %s, \n %s \n %s \n and %s" %\
      (roiPath2,roiPath3,roiPath4,roiPath5)

mp.show()
