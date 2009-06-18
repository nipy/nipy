import numpy as np

from os.path import exists, abspath, join as pjoin
from nipy.io.api import load_image, save_image
from nipy.core import api

from fiac_example import datadir

avganat = load_image(pjoin(datadir, 'group', 'avganat.nii'))
tmap = load_image(pjoin(datadir, 'group', 'block', 'sentence_0', 't.nii'))

import enthought.mayavi.mlab as ML
anat_iso = ML.contour3d(np.array(avganat), opacity=0.4)
tmap_iso = ML.contour3d(np.array(tmap), color=(0.8,0.3,0.3))
