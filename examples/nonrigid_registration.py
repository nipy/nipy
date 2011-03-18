# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from os.path import join
import time

import numpy as np
### DEBUG
from numpy.testing import * 

from nipy import load_image, save_image
from nipy.algorithms.registration import *
from nipy.utils import example_data

from nipy.algorithms.registration.polyaffine import PolyAffine

def debug_resample(Tv, I, J):
    Tv = Affine(inverse_affine(J.affine)).compose(Tv)
    coords = np.indices(I.shape).transpose((1,2,3,0))
    coords = np.reshape(coords, (np.prod(I.shape), 3))
    coords = Tv.apply(coords).T
    return coords

# Input images are provided with the nipy-data package
source = 'ammon'
target = 'anubis'
source_file = example_data.get_filename('neurospin','sulcal2000','nobias_'+source+'.nii.gz')
target_file = example_data.get_filename('neurospin','sulcal2000','nobias_'+target+'.nii.gz')

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
###ca = debug_resample(Av, I, J) 

# Region matching 
t0 = time.time()
dims = np.array(I.shape)/5
corners = []
affines = []
for cz in (1,2,3):
    for cy in (1,2,3): 
        for cx in (1,2,3): 
            corner = np.array((cx,cy,cz))*dims
            print('Doing block: %s' % corner) 
            Ar = A.copy() 
            R.subsample(corner=corner, size=dims)
            R.optimize(Ar)
            corners.append(corner)
            affines.append(Ar) 

# Create polyaffine transform 
t1 = time.time()
centers = np.array(corners) + (dims-1)/2.
### centers = apply_affine(I.affine, centers) 
"""
T = PolyAffine(centers, [Ar.compose(A.inv()) for Ar in affines], 
               dims, glob_affine=A.compose(Affine(I.affine)))
"""

affines = [Ar.compose(Affine(I.affine)) for Ar in affines]
Tv = PolyAffine(centers, affines, .5*dims)

# Resample target image 
t2 = time.time()
Jt = resample(J, Tv, reference=I, ref_voxel_coords=True)
###c = debug_resample(Tv, I, J) 

# Save resampled image
t3 = time.time() 
print('Block-matching time: %f' % (t1-t0)) 
print('Polyaffine creation time: %f' % (t2-t1))
print('Resampling time: %f' % (t3-t2))
save_image(Jt, 'deform_anubis_to_ammon.nii')

