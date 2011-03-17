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


# Input images are provided with the nipy-data package
source = 'ammon'
target = 'anubis'
source_file = example_data.get_filename('neurospin','sulcal2000','nobias_'+source+'.nii.gz')
target_file = example_data.get_filename('neurospin','sulcal2000','nobias_'+target+'.nii.gz')

# Optional arguments
similarity = 'crl1' 
interp = 'pv'
optimizer = 'powell'

# Make registration instance
I = load_image(source_file)
J = load_image(target_file)
R = HistogramRegistration(I, J, similarity=similarity, interp=interp)

# Global affine registration 
A = Affine() 
R.optimize(A)

# Region matching 
t0 = time.time()
dims = np.array(I.shape)/5
corners = []
affines = []
for cz in (1,3):
    for cy in (1,3): 
        for cx in (1,3): 
            corner = np.array((cx,cy,cz))*dims
            print('Doing block: %s' % corner) 
            Ar = A.copy() 
            R.subsample(corner=corner, size=dims)
            R.optimize(Ar)
            corners.append(corner)
            affines.append(Ar) 
            
dt = time.time()-t0 
print('Block matching time: %f' % dt) 

centers = np.array(corners) + (dims-1)/2.
centers = apply_affine(I.affine, centers) 

# Resample target image 
T = PolyAffine(centers, affines, 100., A)
Jt = resample(J, T, reference=I)
save_image(Jt, 'deform_anubis_to_ammon.nii')


"""
0-1*2-3*4-5
"""



