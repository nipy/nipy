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

import os.path as op
import numpy as np
import tempfile

from nipy.io.imageformats import load, save, Nifti1Image
import nipy.neurospin.graph.field as ff
import nipy.neurospin.spatial_models.hroi as hroi
from nipy.neurospin.spatial_models.discrete_domain import domain_from_image

# data paths
import get_data_light
#get_data_light.get_it()

data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data'))
input_image = op.join(data_dir,'spmT_0029.nii.gz')
swd = tempfile.mkdtemp()

# parameters
threshold = 3.0 # blob-forming threshold
smin = 5 # size threshold on bblobs

# prepare the data
nim = load(input_image)
mask_image = Nifti1Image(nim.get_data()**2>0, nim.get_affine())
domain = domain_from_image(mask_image)
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
wlabel = -2*np.ones(nim.get_shape())
wlabel[data!=0] = nroi.label
wim = Nifti1Image(wlabel, nim.get_affine())
wim.get_header()['descrip'] = 'blob image extracted from %s'%input_image 
save(wim,op.join(swd,"blob.nii"))

# saving the image of the average-signal-per-blob
wlabel = np.zeros(nim.get_shape())
wlabel[data!=0] = bmap
wim = Nifti1Image(wlabel, nim.get_affine())
wim.get_header()['descrip'] = 'blob average signal extracted from %s' \
                              %input_image 
save(wim,op.join(swd,"bmap.nii"))

# saving the image of the end blobs or leaves
lroi = nroi.reduce_to_leaves()

wlabel = -2*np.ones(nim.get_shape())
wlabel[data!=0] = lroi.label
wim = Nifti1Image(wlabel, nim.get_affine())
wim.get_header()['descrip'] = 'blob image extracted from %s'%input_image  
save(wim,op.join(swd,"leaves.nii"))

print "Wrote the blob image in %s" %op.join(swd,"blob.nii")
print "Wrote the blob-average signal image in %s" %op.join(swd,"bmap.nii")
print "Wrote the end-blob image in %s" %op.join(swd,"leaves.nii")
