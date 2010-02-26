import numpy as np

from nipy.neurospin.registration import *
from nipy.neurospin.registration.grid_transform import *
from nipy.neurospin.image import *

from nipy.utils import example_data
from nipy.io.imageformats import load as load_image, save as save_image

from os.path import join
import time

print('Scanning data directory...')

# Input images are provided with the nipy-data package
source = 'ammon'
target = 'anubis'
source_file = example_data.get_filename('neurospin','sulcal2000','nobias_'+source+'.nii.gz')
target_file = example_data.get_filename('neurospin','sulcal2000','nobias_'+target+'.nii.gz')

# Optional arguments
similarity = 'cr' 
interp = 'pv'
optimizer = 'powell'

# Make registration instance
I = from_brifti(load_image(source_file))
J = from_brifti(load_image(target_file))
R = IconicRegistration(I, J)
R.set_source_fov(fixed_npoints=64**3)

# Make Gaussian spline transform instance
slices = [slice(0,s.stop,s.step*4) for s in R._slices]
cp = np.mgrid[slices]
cp = np.rollaxis(cp, 0, 4)

T = SplineTransform(I, cp, sigma=5., grid_coords=True)
###T = Ts[R._slices]

# Test 
s = R.eval(T)
sa = R.eval(T.affine)

R.optimize(T, method='conjugate_gradient')
