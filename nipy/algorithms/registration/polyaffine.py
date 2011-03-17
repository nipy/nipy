# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from .transform import Transform
from .affine import inverse_affine, apply_affine
from .c_bindings import _apply_polyaffine

def kernel_matrix(xyz, centers, sigma): 
    """
    Compute covariance matrix
    
    Kij = g(xi-cj) 
    """
    dim = centers.shape[1]
    D = np.zeros((xyz.shape[0], dim, centers.shape[0]))
    D -= centers.T 
    Dt = np.transpose(D, (2,1,0))
    Dt += xyz.T 
    return np.exp(-.5*np.sum(D**2, 1)/sigma**2) 


class PolyAffine(Transform): 

    def __init__(self, centers, affines, sigma, glob_affine=None): 
        """
        centers: N times 3 array 

        We are given a set of affine transforms Ti with centers xi,
        and a global affine Tg. The polyaffine transform is defined,
        up to a right composition with a global affine, as:

        T(x) = sum_i g(x-xi) Qi x

        where Qi = K^-1 Ti.Tg^-1, g is the isotropic Gaussian kernel
        with standard deviation `sigma`, and K is the matrix with
        general element Kij = g(xi-xj).

        Therefore, we have: 
        
        sum_j g(xi-xj) Qj = Ti, for all i 
        
        => T(x) = sum_j g(xi-xj) Qj 
        """

        # Format input arguments
        centers = np.asarray(centers, dtype='double', order='C')
        sigma = float(sigma) 
        if hasattr(affines[0], 'as_affine'):
            affines = np.array([a.as_affine() for a in affines]) 
        else:
            affines = np.asarray(affines)
        if glob_affine == None: 
            glob_affine = np.eye(4) 
        elif hasattr(glob_affine, 'as_affine'):
            glob_affine = glob_affine.as_affine()
        else: 
            glob_affine = np.asarray(glob_affine)

        # Correct local affines for overlapping kernels
        T = np.dot(affines, inverse_affine(glob_affine))
        T = T[:, 0:3, :]
        K = kernel_matrix(centers, centers, sigma)
        L = np.linalg.cholesky(K) # K = L L.T
        Linv = np.linalg.inv(L) 
        Kinv = np.dot(Linv.T, Linv) 
        Q = np.dot(T.T, Kinv.T).T

        # Cache some stuff 
        self.centers = centers
        self.sigma = sigma 
        self.affines = np.reshape(Q[:,0:3,:], (len(self.centers), 12))

        # debug
        self.K = K
        self.Kinv = Kinv 
        self.T = T         


    def apply(self, xyz): 
        """
        xyz is an (N, 3) array 
        """
        txyz = np.array(xyz, copy=True, dtype='double', order='C')
        _apply_polyaffine(txyz, self.centers, self.affines, self.sigma) 
        return txyz


#def compose(self, other): 



"""

x : N x 3 x N 

x += centers.T 
y = np.transpose()

x[i, :, j] == centers[j]

"""
