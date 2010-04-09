"""
This is a little demo that simply shows the effect of ward clustering
on an fMRI dataset

Author: Bertrand Thirion, 2010
"""
print __doc__

import numpy as np
import os
import matplotlib.pylab as mp
from nipy.io.imageformats import load, save, Nifti1Image 
import nipy.neurospin.graph.field as ff
import get_data_light
import tempfile
get_data_light.getIt()

# paths
swd = tempfile.mkdtemp()
data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
InputImage = os.path.join(data_dir,'spmT_0029.nii.gz')
MaskImage = os.path.join(data_dir,'mask.nii.gz')

mask = load(MaskImage).get_data()>0
ijk = np.array(np.where(mask)).T
nvox = ijk.shape[0]
data = load(MaskImage).get_data()[mask]
image_field = ff.Field(nvox)
image_field.from_3d_grid(ijk, k=6)
image_field.set_field(data)
u = image_field.ward(100)

label_image = os.path.join(swd, 'label.nii')
wdata = mask.copy()-1
wdata[mask] = u
save(Nifti1Image(wdata, load(MaskImage).get_affine()), label_image)
print "Label image written in %s" %label_image
