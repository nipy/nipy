# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from _registration import rotation_vec2mat, param_to_vector12, matrix44, affines, _affines
from nipy.neurospin.image import apply_affine

import numpy as np


# Globals 
naffines = 3
id_rigid = affines.index('rigid')
id_similarity = affines.index('similarity')
id_affine = affines.index('affine')

# Defaults 
_radius = 100
_flag2d = False



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


          |   1 - 2y^2 - 2z^2   2xy - 2zw          2xz+ 2yw         |
      R = |   2xy + 2zw         1 - 2x^2 - 2z^2    2yz - 2xw	    |
          |   2xz - 2 yw        2yz + 2xw          1 - 2x^2 - 2y^2  |

	
    """
    TINY = 1e-15

    # Compute the trace of the rotation matrix plus one
    aux = np.sqrt(R.trace()+1.0)
    
    if aux > TINY: 

        # Compute the associated quaternion. Notice: trace(R) + 1 = 4w^2
        quat = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], .5*aux])
        quat[0:3] *= .5/aux
    
        # Compute the angle between 0 and PI (ensure that the last
        # quaternion element is in the range (-1,1))
        theta = 2*np.arccos(max(-1., min(quat[3], 1.)))

        # Normalize the rotation axis
        norma = max(np.sqrt((quat[0:3]**2).sum()), TINY)
        return (theta/norma)*quat[0:3]
    
    else: 
        
        # Singularity case: theta == PI. In this case, the above
        # identification is not possible since w=0. 
        x2 = .25*(1 + R[0][0]-R[1][1]-R[2][2])
        if x2 > TINY: 
            xy = .5*R[1][0]
            xz = .5*R[2][0]
            n = np.array([x2,xy,xz])
        else: 
            y2 = .25*(1 + R[1][1]-R[0][0]-R[2][2])
            if y2 > TINY: 
                xy = .5*R[1][0]
                yz = .5*R[2][1]
                n = np.array([xy,y2,yz])
            else: 
                z2 = .25*(1 + R[2][2]-R[0][0]-R[1][1])
                if z2 > TINY: 
                    xz = .5*R[2][0]
                    yz = .5*R[2][1]
                    n = np.array([xz,yz,z2])
        return np.pi*n/np.sqrt((n**2).sum())


def vector12(mat, subtype=id_affine):
    """
    Return a 12-sized vector of natural affine parameters:
    translation, rotation, log-scale, additional rotation (to allow
    for shearing when combined with non-unitary scales). 
    """
    TINY = 1e-100
    vec12 = np.zeros(12)
    vec12[0:3] = mat[0:3,3]
    A = mat[0:3,0:3]
    if subtype == id_rigid: 
        vec12[3:6] = rotation_mat2vec(A)
        vec12[6:9] = 0.0
    elif subtype == id_similarity:
        ## A = s R ==> det A = (s)**3 ==> s = (det A)**(1/3)
        s = np.linalg.det(A)**(1/3.)
        vec12[3:6] = rotation_mat2vec(A/s)
        vec12[6:9] = np.log(np.maximum(s, TINY))
    else: 
        R, s, Q = np.linalg.svd(mat[0:3,0:3]) # mat == R*diag(s)*Q
        if np.linalg.det(R) < 0: 
            R = -R
            Q = -Q
        r = rotation_mat2vec(R)
        q = rotation_mat2vec(Q)
        vec12[3:6] = r
        vec12[6:9] = np.log(np.maximum(s, TINY))
        vec12[9:12] = q
    return vec12

    

def preconditioner(radius):
    """
    Computes a scaling vector pc such that, if p=(u,r,s,q) represents
    affine transformation parameters, where u is a translation, r and
    q are rotation vectors, and s is the vector of log-scales, then
    all components of (p/pc) are roughly comparable to the translation
    component.

    To that end, we use a `radius` parameter which represents the
    'typical size' of the object being registered. This is used to
    reformat the parameter vector
    (translation+rotation+scaling+shearing) so that each element
    roughly represents a variation in mm.
    """
    rad = 1./radius
    sca = 1./radius
    return np.array([1,1,1,rad,rad,rad,sca,sca,sca,rad,rad,rad])


class Affine(object): 

    def __init__(self, array=None, radius=_radius, flag2d=_flag2d):
        subtype = id_affine
        self._generic_init(array, radius, subtype, flag2d)
    
    def _generic_init(self, array, radius, subtype, flag2d): 
        if array == None: 
            vec12 = np.zeros(12)
        elif array.shape == (4,4):
            vec12 = vector12(array)
        elif array.size == 12: 
            vec12 = array.ravel()
        else: 
            raise ValueError('Invalid array')
        self._set_vec12(vec12)
        self._precond = preconditioner(radius)
        self._subtype = subtype
        self._flag2d = flag2d
        self._stamp = subtype + naffines*(not flag2d)


    def __call__(self, xyz): 
        return apply_affine(self.__array__(), xyz)

    def _get_param(self): 
        param = self._vec12/self._precond
        return param[_affines[self._stamp]]

    def _set_param(self, p): 
        self._vec12 = param_to_vector12(np.asarray(p), self._vec12, 
                                        self._precond, self._stamp)
        
    param = property(_get_param, _set_param)

    def _get_vec12(self):
        return self._vec12

    def _set_vec12(self, vec12): 
        # Specify dtype to allow in-place operations
        self._vec12 = np.asarray(vec12, dtype='double') 
        
    vec12 = property(_get_vec12, _set_vec12)

    def _get_precond(self): 
        return self._precond 

    precond = property(_get_precond)

    def __array__(self, dtype='double'): 
        return matrix44(self._vec12, dtype=dtype)

    def __str__(self): 
        string  = 'translation : %s\n' % self._vec12[0:3].__str__()
        string += 'rotation    : %s\n' % self._vec12[3:6].__str__()
        string += 'scaling     : %s\n' % (np.exp(self._vec12[6:9])).__str__()
        string += 'shearing    : %s' % self._vec12[9:12].__str__()
        return string

    def __mul__(self, other): 
        """
        Affine composition: T1oT2(x)
        """
        subtype = max(self._subtype, other._subtype)
        a = affine_classes[subtype]()
        a._precond = self._precond
        a._set_vec12(vector12(np.dot(self.__array__(), other.__array__()), a._subtype))
        return a

    def inv(self):
        """
        Return the inverse affine transform. 
        """
        a = affine_classes[self._subtype]()
        a._precond = self._precond
        a._set_vec12(vector12(np.linalg.inv(self.__array__()), self._subtype))
        return a
        

class Rigid(Affine):

    def __init__(self, array=None, radius=_radius, flag2d=_flag2d):
        subtype = id_rigid
        self._generic_init(array, radius, subtype, flag2d)
        
    def __str__(self): 
        string  = 'translation : %s\n' % self._vec12[0:3].__str__()
        string += 'rotation    : %s\n' % self._vec12[3:6].__str__()
        return string

class Similarity(Affine):

    def __init__(self, array=None, radius=_radius, flag2d=_flag2d):
        subtype = id_similarity
        self._generic_init(array, radius, subtype, flag2d)

    def __str__(self): 
        string  = 'translation : %s\n' % self._vec12[0:3].__str__()
        string += 'rotation    : %s\n' % self._vec12[3:6].__str__()
        string += 'scaling     : %s\n' % (np.exp(self._vec12[6])).__str__()
        return string
    

# List of 
affine_classes = [Rigid, Similarity, Affine]

