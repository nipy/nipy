from routines import rotation_vec2mat, param_to_vector12, matrix44, affines, _affines

import numpy as np



def rotation_mat2vec(R):
    """
    r = rotation_mat2vec(R)
    
    Inverse operation of rotation_vec2mat. 
    
    The algorithm is based on a quaternion representation, exploiting the
    fact that the rotation vector r = theta*n associated with a quaternion
    (x,y,z,w) statisfies:
    
      x = sin(theta/2) nx
      y = sin(theta/2) ny
      z = sin(theta/2) nz
      w = cos(theta/2)
    """
    TINY = 1e-15

    # Compute the trace of the rotation matrix plus one
    aux = np.sqrt(np.maximum(R.trace()+1.0, TINY))
    
    # Compute the associated quaternion. Notice: trace(R) + 1 = 4w^2
    quat = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], .5*aux])
    quat[0:3] *= .5/aux
	
    # Compute the angle between 0 and PI
    if np.abs(quat[3])<1:
        theta = 2*np.arccos(quat[3])
    else: 
        return np.zeros(3)

    # Normalize r
    return theta/np.sqrt((quat[0:3]**2).sum())


def vector12(mat): 
    R, s, Q = np.linalg.svd(mat[0:3,0:3])
    r = rotation_mat2vec(R)
    q = rotation_mat2vec(Q)
    vec12 = np.zeros(12)
    vec12[0:3] = mat[0:3,3]
    vec12[3:6] = r
    vec12[6:9] = s
    vec12[9:12] = q
    return vec12

    

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


class Affine: 

    def __init__(self, subtype='affine', vec12=None, radius=1, flag3d=True):
        self.precond = preconditioner(radius)
        self._subtype = affines.index(subtype)+len(affines)*flag3d
        if vec12 == None: 
            vec12 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])
        self.set_vec12(vec12)

    def __call__(self, xyz): 
        return apply(xyz, self.mat44())


    def subtype(self): 
        subtype = affines[self._subtype%len(affines)]
        if self._subtype/len(affines)==0: 
            dim = '2d'
        else: 
            dim = '3d'
        return subtype, dim

    def from_param(self, p): 
        self.vec12 = param_to_vector12(np.asarray(p), self.vec12, self.precond, self._subtype)
        
    def to_param(self): 
        param = self.vec12/self.precond
        return param[_affines[self._subtype]]
        
    def set_vec12(self, vec12): 
        # Specify dtype to allow in-place operations
        self.vec12 = np.asarray(vec12, dtype='double') 
        
    def mat44(self): 
        return matrix44(self.vec12)

    def __str__(self): 
        return self.vec12[_affines[self._subtype]].__str__()

    def __mul__(self, other): 
        """
        Affine composition: T1oT2(x)
        """
        vec12 = vector12(np.dot(self.mat44(), other.mat44()))
        a = Affine(vec12=vec12)
        a._subtype = max(self._subtype, other._subtype)
        a.precond = self.precond
        return a
