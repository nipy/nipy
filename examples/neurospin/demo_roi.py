"""
This is a little demo that simply shows ROI manipulation within
the nipy framework

Author: Bertrand Thirion, 2009
"""
import numpy as np
import os
import nifti
from nipy.neurospin.utils.roi import ROI
import get_data_light
import tempfile
get_data_light.getIt()

# paths
swd = tempfile.mkdtemp()
data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
InputImage = os.path.join(data_dir,'spmT_0029.nii.gz')
MaskImage = os.path.join(data_dir,'mask.nii.gz')

# create the ROI
position = [0,0,0]
nim = nifti.NiftiImage(MaskImage)
header = nim.header
dat = nim.asarray().T
roi = ROI("myroi",header)
roi.from_position(np.array(position),5.0)
roi.make_image(os.path.join(swd,"myroi.nii"))
roi.set_feature_from_image('activ',InputImage)
roi.plot_feature('activ')

print 'Wrote an ROI mask image in %s' %os.path.join(swd,"myroi.nii")

import matplotlib.pylab as mp
mp.show()
