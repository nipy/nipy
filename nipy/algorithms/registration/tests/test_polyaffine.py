import numpy as np 

from nipy.algorithms.registration.polyaffine import *


#xyz = np.random.rand(19, 3) 
#centers = np.random.rand(10, 3) 
#K = kernel_matrix(xyz, centers, 1.0)

def random_affine():
    T = np.eye(4) 
    T[0:3,0:4] = np.random.rand(3, 4)
    return T 

def id_affine():
    return np.eye(4) 


centers = [np.random.rand(3) for i in range(5)]
affines = [id_affine() for i in range(5)]
sigma = .0001

T = PolyAffine(centers, affines, sigma) 

xyz = np.random.rand(2, 3) 
t = T.apply(xyz) 
t2 = T.apply2(xyz) 


#zoo = T.apply(np.asarray(centers))

