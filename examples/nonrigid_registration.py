import numpy as np

from nipy.neurospin.registration import *
from nipy.neurospin.registration.grid_transform import *
from nipy.neurospin.image import *

from nipy.utils import example_data
from nipy.io.imageformats import load as load_image, save as save_image

### DEBUG
from numpy.testing import * 

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

# Start with an affine registration
A0 = Affine()
###A = R.optimize(A0)
A = Affine()

# Then add control points...
T0 = SplineTransform(I, cp, sigma=5., grid_coords=True, affine=A)

# Test 1
s = R.eval(T0)
sa = R.eval(T0.affine)
assert_almost_equal(s, sa)

# Test 2
T = SplineTransform(I, cp, sigma=5., grid_coords=True, affine=A)
T0v = T0[R._slices]()
Tv = T[R._slices]()
assert_almost_equal(Tv, T0v)

# Test 3
T.param += 1.
s0 = R.eval(T0)
s = R.eval(T)

T = R.optimize(T0, method='conjugate_gradient', gtol=.01)

###
t = T()
Jt = transform_image(J, t, 'grid', reference=I)

###Jt = transform(to_brifti(J), T, reference=to_brifti(I))


###save_image(to_brifti(Jt), 'deform_anubis_to_ammon.nii')
