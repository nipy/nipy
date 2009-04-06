from routines import rotation_vector_to_matrix, param_to_vector12, matrix44, affine_types, affine_indices

import numpy as np


def preconditioner(radius):
    """
    Computes a scaling vector pc such that, if p=(u,r,s,q) represents
    affine transformation parameters, where u is a translation, r and
    q are rotation vectors, and s is a scaling vector, then all
    components of (p/pc) are somehow comparable and homogeneous to the
    distance unit implied by the translation component.
    """
    rad = 1./radius
    sca = radius
    return np.array([1,1,1,rad,rad,rad,sca,sca,sca,rad,rad,rad])


def apply(xyz, T):
    """
    XYZ = transform(xyz, T)

    T is a 4x4 matrix.
    xyz is a 3xN array of 3d coordinates stored row-wise.  
    """
    XYZ = np.dot(T[0:3,0:3], xyz)
    XYZ[0,:] += T[0,3]
    XYZ[1,:] += T[1,3]
    XYZ[2,:] += T[2,3]
    return XYZ 


class AffineTransform: 

    def __init__(self, subtype='affine', vec12=None, radius=1, flag2d=False):
        self.precond = preconditioner(radius)
        self.subtype = subtype
        if flag2d: 
            self.dim = '2d'
        else: 
            self.dim = '3d'
        self._subtype = affine_types[self.dim][subtype]
        if vec12 == None: 
            vec12 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])
        self.set_vec12(vec12)


    def __call__(self, xyz): 
        return apply(xyz, self.mat44)

    def from_param(self, p): 
        self.vec12 = param_to_vector12(np.asarray(p), self.vec12, self.precond, self._subtype)
        self.mat44 = matrix44(self.vec12)

    def to_param(self): 
        param = self.vec12/self.precond
        return param[affine_indices[self.dim][self.subtype]]
        
    def set_vec12(self, vec12): 
        # Specify dtype to allow in-place operations
        self.vec12 = np.asarray(vec12, dtype='double') 
        self.mat44 = matrix44(self.vec12)

    def __str__(self): 
        s =  '  translation : %s\n' % self.vec12[0:3].__str__()
        s += '  rotation    : %s\n' % self.vec12[3:6].__str__()
        s += '  scaling     : %s\n' % self.vec12[6:9].__str__()
        s += '  shearing    : %s' % self.vec12[9:12].__str__()
        return s
