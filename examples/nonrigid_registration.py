# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import time

import numpy as np

from nipy import load_image, save_image
from nipy.utils import example_data
from nipy.algorithms.registration import (Affine,
                                          HistogramRegistration,
                                          resample)
from nipy.algorithms.registration.polyaffine import PolyAffine


def get_blocks(shape, segments, step, skip):
    skipf = segments - skip
    size = np.array(shape) / segments
    tmp = np.mgrid[skip:skipf:step, skip:skipf:step, skip:skipf:step]
    tmp = np.reshape(tmp, (3, np.prod(tmp.shape[1:]))).T
    return tmp * size, size


# Input images are provided with the nipy-data package
source = 'ammon'
target = 'anubis'
source_file = example_data.get_filename('neurospin', 'sulcal2000',
                                        'nobias_' + source + '.nii.gz')
target_file = example_data.get_filename('neurospin', 'sulcal2000',
                                        'nobias_' + target + '.nii.gz')

# Optional arguments
similarity = 'cc'
interp = 'pv'
optimizer = 'powell'

# Make registration instance
I = load_image(source_file)
J = load_image(target_file)
R = HistogramRegistration(I, J, similarity=similarity, interp=interp)

# Global affine registration
A = Affine()
R.optimize(A)

#Jt = resample(J, A, reference=I)
Av = A.compose(Affine(I.affine))
Jat = resample(J, Av, reference=I, ref_voxel_coords=True)
save_image(Jat, 'affine_anubis_to_ammon.nii')

# Region matching
t0 = time.time()

##corners, size = get_blocks(I.shape, 3, 1, 0) #.5 size
##corners, size = get_blocks(I.shape, 6, 2, 0) #.75 size
##corners, size = get_blocks(I.shape, 6, 1, 0) # .5 size

corners, size = get_blocks(I.shape, 5, 2, 1)

affines = []
for corner in corners:
    print('Doing block: %s' % corner)
    Ar = A.copy()
    R.subsample(corner=corner, size=size)
    R.optimize(Ar)
    affines.append(Ar)

# Create polyaffine transform
t1 = time.time()
centers = np.array(corners) + (size - 1) / 2.
affines = [Ar.compose(Affine(I.affine)) for Ar in affines]
Tv = PolyAffine(centers, affines, .5 * size)

# Resample target image
t2 = time.time()
Jt = resample(J, Tv, reference=I, ref_voxel_coords=True)
###c = debug_resample(Tv, I, J)

# Save resampled image
t3 = time.time()
print('Block-matching time: %f' % (t1 - t0))
print('Polyaffine creation time: %f' % (t2 - t1))
print('Resampling time: %f' % (t3 - t2))
save_image(Jt, 'deform_anubis_to_ammon.nii')
