import fff2.neuro
import numpy as np

from datamind.core import DF

fmri_dataset_path = '/neurospin/lnao/Panabase/data_fiac/fiac_fsl/fiac0/fMRI/acquisition/fonc1/afonc1.nii.gz'
design_matrix_path = '/neurospin/lnao/Panabase/data_fiac/fiac_fsl/fiac0/fMRI/acquisition/glm/default/fonc1/design_mat.csv'
mask_image_path = '/neurospin/lnao/Panabase/data_fiac/fiac_fsl/fiac0/fMRI/acquisition/Minf/mask.nii'
 
# Get design matrix as numpy array
print('loading design matrix...')
X = np.asarray(DF.read(design_matrix_path))

# Get fMRI data as numpy array
print('loading fmri data...')
Y = fff2.neuro.image(fmri_dataset_path)

# Get the mask
print('loading mask...')
##Mask = fff2.neuro.image(mask_image_path)
Mask = None

# GLM options
##model = 'ar1'
model = 'spherical'
    
# Fit 
print('starting fit...')
glm = fff2.neuro.linear_model(Y, X, Mask, model=model)

# Compute aribtrary contrast image 
print('computing test contrast image...')
c = np.ones(X.shape[1])
c[0] = 1.
con, vcon, dof = glm.contrast(c)
##con.save(whatever_filename_you_like)
               
