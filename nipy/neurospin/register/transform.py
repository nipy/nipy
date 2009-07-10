from routines import rotation_vec2mat, param_to_vector12, matrix44, affines, _affines


import numpy as np

# Some constants 
BRAIN_RADIUS_MM = 10 ## used for pre-conditioning in optimization context


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


def vector12(mat, subtype='affine'):
    vec12 = np.zeros(12)
    vec12[0:3] = mat[0:3,3]
    A = mat[0:3,0:3]
    if subtype == 'rigid': 
        vec12[3:6] = rotation_mat2vec(A)
        vec12[6:9] = 1
    elif subtype == 'similarity':
        ## A = s R ==> det A = (s)**3 ==> s = (det A)**(1/3)
        s = np.linalg.det(A)**(1/3.)
        vec12[3:6] = rotation_mat2vec(A/s)
        vec12[6:9] = s
    else: 
        R, s, Q = np.linalg.svd(mat[0:3,0:3]) # mat == R*diag(s)*Q
        if np.linalg.det(R) < 0: 
            R = -R
            Q = -Q
        r = rotation_mat2vec(R)
        q = rotation_mat2vec(Q)
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


def apply_affine(T, xyz):
    """
    XYZ = apply_affine(T, xyz)

    T is a 4x4 matrix.
    xyz is a 3xN array of 3d coordinates stored row-wise.  
    """
    XYZ = np.dot(T[0:3,0:3], xyz)
    XYZ[0,:] += T[0,3]
    XYZ[1,:] += T[1,3]
    XYZ[2,:] += T[2,3]
    return XYZ 


class Affine(): 

    def __init__(self, subtype='affine', vec12=None, radius=1, flag3d=True):
        self.precond = preconditioner(radius)
        self._subtype = affines.index(subtype)+len(affines)*flag3d
        if vec12 == None: 
            vec12 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])
        self.set_vec12(vec12)

    def __call__(self, xyz): 
        return apply_affine(self.__array__(), xyz)

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
        
    def __array__(self, dtype='double'): 
        return matrix44(self.vec12, dtype=dtype)

    def __str__(self): 
        str  = 'translation : %s\n' % self.vec12[0:3].__str__()
        str += 'rotation    : %s\n' % self.vec12[3:6].__str__()
        str += 'scaling     : %s\n' % self.vec12[6:9].__str__()
        str += 'shearing    : %s' % self.vec12[9:12].__str__()
        return str
        #return self.vec12[_affines[self._subtype]].__str__()

    def __mul__(self, other): 
        """
        Affine composition: T1oT2(x)
        """
        a = Affine()
        a._subtype = max(self._subtype, other._subtype)
        a.precond = self.precond
        a.set_vec12(vector12(np.dot(self.__array__(), other.__array__()), a.subtype()[0]))
        return a


