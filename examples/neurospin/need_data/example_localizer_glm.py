"""
Example of (step-by-step) GLM application and result creation.
More specifically,
1. A sequence of fMRI volumes are loaded
2. A design matrix describing all the effects related to the data is computed
3. A GLM is applied to the dataset

Note that this corresponds to a single session

Author : Bertrand Thirion, 2010
"""
print __doc__

import numpy as np
import os.path as op
import matplotlib.pylab as mp
import pylab

from nipy.neurospin.utils.mask import compute_mask_files
from nipy.io.imageformats import load, save, Nifti1Image
import get_data_light
import nipy.neurospin.glm
import nipy.neurospin.utils.design_matrix as dm
import tempfile

#######################################
# Data and analysis parameters
#######################################

# volume mask
get_data_light.get_localizer_dataset()
data_path = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 's12069_swaloc1_corr.nii.gz'))
paradigm_file = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                      'localizer_paradigm.csv'))

# timing
n_scans = 128
tr = 2.4

# paradigm
frametimes = np.linspace(0, (n_scans-1)*tr, n_scans)
conditions = [ 'damier_H', 'damier_V', 'clicDaudio', 'clicGaudio', 
'clicDvideo', 'clicGvideo', 'calculaudio', 'calculvideo', 'phrasevideo', 
'phraseaudio' ]

# confounds
hrf_model = 'Canonical With Derivative'
drift_model = "Cosine"
hfcut = 128

# write directory
swd = tempfile.mkdtemp()

########################################
# Design matrix
########################################

paradigm = dm.load_protocol_from_csv_file(paradigm_file, session=0)

design_matrix = dm.DesignMatrix( frametimes, paradigm, hrf_model=hrf_model,
                                 drift_model=drift_model, hfcut=hfcut,
                                 cond_ids= conditions)

design_matrix.show()
# design_matrix.save(...)

########################################
# Mask the data
########################################

mask_path = op.join(swd, 'mask.nii') 
mask_array = compute_mask_files( data_path, mask_path, True, 0.4, 0.9)>0

########################################
# Perform a GLM analysis
########################################

fmri_image = load(data_path)
Y = fmri_image.get_data()[mask_array]
model = "ar1"
method = "kalman"
my_glm = nipy.neurospin.glm.glm()
glm = my_glm.fit(Y.T, design_matrix.matrix,
                 method="kalman", model="ar1")

#########################################
# Specify the contrasts
#########################################
nc = np.zeros(26)#design_matrix.n_main_regressors)
contrasts = {'damier_H': nc}



#########################################
# Estimate the contrasts
#########################################

for contrast_id in contrasts:
    lcontrast = my_glm.contrast(contrasts[contrast_id])
    contrast_path = op.join(swd, '%s_zmap.nii'% contrast_id)
    save(contrast_image, contrast_path)




#########################################
# End
#########################################

print "All the  results were witten in %s" %swd
pylab.show()
