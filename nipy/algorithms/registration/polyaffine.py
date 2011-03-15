# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from .transform import Transform


class PolyAffine(Transform): 

    def __init__(self, centers, affines, sigma, glob_affine=None): 
        """
        centers: N times 3 array 

        We are given a set of affine transforms Ti with centers xi,
        and a global affine Tg. The polyaffine transform is defined
        up to a right composition with a global affine as:

        T(x) = x + sum_i g(x-xi) Qi x

        where Qi = G^-1 (Ti.Tg^-1-Id), g is the isotropic Gaussian kernel with
        standard deviation `sigma`, and G is the matrix with general
        element Gij = g(xi-xj). 

        Therefore, we have: 
        
        sum_j g(xi-xj) Qj = Ti, for all i 
        
        => T(x) = sum_j g(xi-xj) Qj 
        """

        # Compute covariance matrix
        nc = centers.shape[0] 
        d = centers.shape[1]
        D = np.zeros((nc, d, nc))
        D -= centers.T 
        Dt = np.transpose(D, (2,1,0))
        Dt += centers.T 
        G = np.exp(-.5*np.sum(D**2, 1)/sigma**2) 
        L = np.linalg.cholesky(G) # G = L L.T
        Linv = np.linalg.inv(L) 
        Ginv = np.dot(Linv.T, Linv) 

        # Compute 
        if hasattr(affines[0], 'as_affine'):
            y = np.array([a.as_affine() for a in affines]) 
        else:
            y = np.asarray(affines) 
        x = np.dot((y-np.eye(d+1)).T,Ginv.T).T

        # cache some stuff to debug 
        self.G = G 
        self.Ginv = np.dot(Linv.T, Linv) 

        
        

    #def apply(self, xyz): 
    #def compose(self, other): 



"""

x : N x 3 x N 

x += centers.T 
y = np.transpose()

x[i, :, j] == centers[j]

"""
