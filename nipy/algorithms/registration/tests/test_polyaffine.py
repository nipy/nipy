import numpy as np 

from nipy.algorithms.registration.polyaffine import *
from nipy.algorithms.registration.c_bindings import _apply_polyaffine

#xyz = np.random.rand(19, 3) 
#centers = np.random.rand(10, 3) 
#K = kernel_matrix(xyz, centers, 1.0)

def random_affine():
    T = np.eye(4) 
    T[0:3,0:4] = np.random.rand(3, 4)
    return T 

def id_affine():
    return np.eye(4) 


NCENTERS = 5
NPTS = 100

centers = [np.random.rand(3) for i in range(NCENTERS)]
affines = [random_affine() for i in range(NCENTERS)]
sigma = 1.
xyz = np.random.rand(NPTS, 3) 
#xyz = np.array(centers)


T = PolyAffine(centers, affines, sigma) 
t = T.apply(xyz) 

"""
txyz = xyz.copy()

affines = np.array(affines)
_affines = np.reshape(affines[:,0:3,:], (NCENTERS, 12))

_apply_polyaffine(txyz, centers, _affines, sigma) 
"""


