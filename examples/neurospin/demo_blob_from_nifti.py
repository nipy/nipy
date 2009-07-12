"""
This scipt generates a noisy activation image image
and extracts the blob from it.

Author : Bertrand Thirion, 2009
"""
#autoindent

import os
import numpy as np
import nifti
import nipy.neurospin.graph.field as ff
import nipy.neurospin.spatial_models.hroi as hroi

data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
inputImage = os.path.join(data_dir,'zstat1.nii.gz')
# if it doe not exist, please use nipy/utils/get_data


nim = nifti.NiftiImage(inputImage)
header = nim.header
data = nim.asarray().T
xyz = np.array(np.where(data)).T
F = ff.Field(xyz.shape[0])
F.from_3d_grid(xyz)
F.set_field(data[data!=0])


label = -np.ones(F.V)
nroi = hroi.NROI_from_field(F,header,xyz,0,3.0,smin=10)
if nroi!=None:
    idx = nroi.discrete_features['masked_index']
    for k in range(nroi.k):
        label[idx[k]] = k
        
wlabel = -2*np.ones(nim.getVolumeExtent())
wlabel[data!=0]=label
wim =  nifti.NiftiImage(wlabel.T,nim.header)
wim.description='blob image extracted from %s'%inputImage 
wim.save("/tmp/blob.nii")

