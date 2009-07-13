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

get_data_light.getIt()

# paths
swd = "/tmp/"
data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
InputImage = os.path.join(data_dir,'spmT_0029.nii.gz')
MaskImage = os.path.join(data_dir,'mask.nii.gz')


nim = nifti.NiftiImage(MaskImage)
header = nim.header
dat = nim.asarray().T
roi = ROI("myroi",header)
roi.from_position(np.array([0,0,0]),5.0)
roi.make_image(op.join(swd,"myroi.nii"))
roi.set_feature_from_image('activ',InputImage)
roi.plot_feature('activ')

import matplotlib.pylab as mp
mp.show()
