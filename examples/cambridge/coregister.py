from nipy.neurospin.registration import *
from nipy import load_image, save_image 
from os.path import join 

DATADIR = '/home/alexis/tmp/fiac0'

anat = load_image(join(DATADIR, 'raw_anatomical.nii'))
func = load_image(join(DATADIR, 'meanafunctional_01.nii'))

R = IconicRegistration(func, anat) 
R.similarity = 'crL1'
R.interp = 'pv'

T = R.optimize(Affine())

# Resample source image
print('Resampling functional image...')
resampled_func = resample(func, T.inv(), reference=anat)

save_image(resampled_func, 'resampled_func.nii')

