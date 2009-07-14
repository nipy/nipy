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

import os.path as op
import numpy as np
import nifti
import tempfile
import nipy.neurospin.graph.field as ff
import nipy.neurospin.spatial_models.hroi as hroi

# data paths
import get_data_light
get_data_light.getIt()
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data'))
inputImage = op.join(data_dir,'spmT_0029.nii.gz')
swd = tempfile.mkdtemp()

# parameters
threshold = 3.0 # blob-forming threshold
smin = 5 # size threshold on bblobs

# prepsare the data
nim = nifti.NiftiImage(inputImage)
header = nim.header
data = nim.asarray().T
xyz = np.array(np.where(data)).T
F = ff.Field(xyz.shape[0])
F.from_3d_grid(xyz)
F.set_field(data[data!=0])

# compute the  nested roi object
label = -np.ones(F.V)
nroi = hroi.NROI_from_field(F,header,xyz,0,threshold,smin)
bfm = nroi.discrete_to_roi_features('activation')
bmap = -np.zeros(F.V)
if nroi!=None:
    idx = nroi.discrete_features['masked_index']
    for k in range(nroi.k):
        label[idx[k]] = k
        bmap[idx[k]] = bfm[k]

# saving the blob image,i. e. a label image 
wlabel = -2*np.ones(nim.getVolumeExtent())
wlabel[data!=0] = label
wim =  nifti.NiftiImage(wlabel.T,nim.header)
wim.description = 'blob image extracted from %s'%inputImage 
wim.save(op.join(swd,"blob.nii"))

# saving the image of the average-signal-per-blob
wlabel = np.zeros(nim.getVolumeExtent())
wlabel[data!=0] = bmap
wim =  nifti.NiftiImage(wlabel.T,nim.header)
wim.description = 'blob image extracted from %s'%inputImage 
wim.save(op.join(swd,"bmap.nii"))

# saving the image of the end blobs or leaves
lroi = nroi.reduce_to_leaves()
label = -np.ones(F.V)
if lroi!=None:
    idx = lroi.discrete_features['masked_index']
    for k in range(lroi.k):
        label[idx[k]] = k
wlabel = -2*np.ones(nim.getVolumeExtent())
wlabel[data!=0] = label
wim =  nifti.NiftiImage(wlabel.T,nim.header)
wim.description = 'blob image extracted from %s'%inputImage 
wim.save(op.join(swd,"leaves.nii"))

print "Wrote the blob image in %s" %op.join(swd,"blob.nii")
print "Wrote the blob-average signal image in %s" %op.join(swd,"bmap.nii")
print "Wrote the end-blob image in %s" %op.join(swd,"leaves.nii")
