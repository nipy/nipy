# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import scipy.linalg as spl

from nipy.externals.transforms3d.quaternions import mat2quat, quat2axangle

from .transform import Transform

# Defaults
_radius = 100

# Smallest possible scaling
TINY = 1e-100

def rotation_mat2vec(R):
    """ Rotation vector from rotation matrix `R`

    Parameters
    ----------
    R : (3,3) array-like
        Rotation matrix

    Returns
    -------
    vec : (3,) array
        Rotation vector, where norm of `vec` is the angle ``theta``, and the
        axis of rotation is given by ``vec / theta``
    """
    ax, angle = quat2axangle(mat2quat(R))
    return ax * angle


def rotation_vec2mat(r):
    """
    R = rotation_vec2mat(r)

    The rotation matrix is given by the Rodrigues formula:
    
    R = Id + sin(theta)*Sn + (1-cos(theta))*Sn^2  
    
    with:
    
           0  -nz  ny
    Sn =   nz   0 -nx
          -ny  nx   0
    
    where n = r / ||r||
    
    In case the angle ||r|| is very small, the above formula may lead
    to numerical instabilities. We instead use a Taylor expansion
    around theta=0:
    
    R = I + sin(theta)/tetha Sr + (1-cos(theta))/teta2 Sr^2
    
    leading to:
    
    R = I + (1-theta2/6)*Sr + (1/2-theta2/24)*Sr^2
    """
    theta = spl.norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
        R = np.eye(3) + np.sin(theta)*Sn + (1-np.cos(theta))*np.dot(Sn,Sn)
    else:
        Sr = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
        theta2 = theta*theta
        R = np.eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*np.dot(Sr,Sr)
    return R


def matrix44(t, dtype=np.double):
    """
    T = matrix44(t)

    t is a vector of of affine transformation parameters with size at
    least 6.

    size < 6 ==> error
    size == 6 ==> t is interpreted as translation + rotation
    size == 7 ==> t is interpreted as translation + rotation + isotropic scaling
    7 < size < 12 ==> error
    size >= 12 ==> t is interpreted as translation + rotation + scaling + shearing 
    """
    size = t.size
    T = np.eye(4, dtype=dtype)
    R = rotation_vec2mat(t[3:6])
    if size == 6:
        T[0:3,0:3] = R
    elif size == 7:
        T[0:3,0:3] = t[6]*R
    else:
        S = np.diag(np.exp(t[6:9])) 
        Q = rotation_vec2mat(t[9:12]) 
        # Beware: R*s*Q
        T[0:3,0:3] = np.dot(R,np.dot(S,Q))
    T[0:3,3] = t[0:3] 
    return T 


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


def inverse_affine(affine):
    return spl.inv(affine)


def apply_affine(aff, pts):
    """ Apply affine `T` to points `pts`

    Parameters
    ----------
    aff : (N, N) array-like
        Homogenous affine, probably 4 by 4
    pts : (P, N-1) array-like
        Points, one point per row.  N-1 is probably 3.

    Returns
    -------
    transformed_pts : (P, N-1) array
        transformed points
    """
    # Apply N by N homogenous affine to P by N-1 array
    pts = np.asarray(pts)
    shape = pts.shape
    pts = pts.reshape((-1, shape[-1]))
    rzs = aff[:-1,:-1]
    trans = aff[:-1,-1]
    res = np.dot(pts, rzs.T) + trans[None,:]
    return res.reshape(shape)


def subgrid_affine(affine, slices):
    steps = map(lambda x: max(x,1), [s.step for s in slices])
    starts = map(lambda x: max(x,0), [s.start for s in slices])
    t = np.diag(np.concatenate((steps,[1]),1))
    t[0:3,3] = starts
    return np.dot(affine, t)


class Affine(Transform): 
    param_inds = range(12)

    def __init__(self, array=None, radius=_radius):
        if array == None: 
            self._vec12 = np.zeros(12)
        elif array.shape == (4,4):
            self._vec12 = self._vector12(array)
        elif array.size == 12: 
            self._vec12 = array.ravel()
        else: 
            raise ValueError('Invalid array')
        self._precond = preconditioner(radius)

    def _vector12(self, aff):
        """
        Return a 12-sized vector of natural affine parameters:
        translation, rotation, log-scale, pre-rotation (to allow for
        shearing when combined with non-unitary scales).

        a better naming is
        vec12=[translation, post-rot, logscaling, pre-rot]
        """
        vec12 = np.zeros((12,))
        vec12[0:3] = aff[:3,3]
        R, s, Q = spl.svd(aff[0:3,0:3]) # mat == R*diag(s)*Q
        if spl.det(R) < 0:
            R = -R
            Q = -Q
        r = rotation_mat2vec(R)
        q = rotation_mat2vec(Q)
        vec12[3:6] = r
        vec12[6:9] = np.log(np.maximum(s, TINY))
        vec12[9:12] = q
        return vec12

    def apply(self, xyz): 
        return apply_affine(self.as_affine(), xyz)

    def _get_param(self): 
        param = self._vec12/self._precond
        return param[self.param_inds]

    def _set_param(self, p):
        p = np.asarray(p)
        inds = self.param_inds
        self._vec12[inds] = p * self._precond[inds]

    def _get_translation(self): 
        return self._vec12[0:3]

    def _set_translation(self, x): 
        self._vec12[0:3] = x

    def _get_rotation(self):
        return self._vec12[3:6]

    def _set_rotation(self, x): 
        self._vec12[3:6] = x

    def _get_scaling(self):
        return np.exp(self._vec12[6:9])

    def _set_scaling(self, x): 
        self._vec12[6:9] = np.log(x)

    def _get_shearing(self): 
        return self._vec12[9:12]

    def _set_shearing(self, x): 
        self._vec12[9:12] = x

    def _get_precond(self): 
        return self._precond 

    translation = property(_get_translation, _set_translation)
    rotation = property(_get_rotation, _set_rotation)
    scaling = property(_get_scaling, _set_scaling)
    shearing = property(_get_shearing, _set_shearing)
    precond = property(_get_precond)
    param = property(_get_param, _set_param)

    def as_affine(self, dtype='double'): 
        return matrix44(self._vec12, dtype=dtype)

    def compose(self, other):
        """ Compose this transform onto another

        Parameters
        ----------
        other : Transform
            transform that we compose onto

        Returns
        -------
        composed_transform : Transform
            a transform implementing the composition of self on `other`
        """
        try:
            other_aff = other.as_affine()
        except AttributeError:
            return Transform(self.apply).compose(other)
        # Choose more capable of input types as output type
        self_inds = set(self.param_inds)
        other_inds = set(other.param_inds)
        if self_inds.issubset(other_inds):
            klass = other.__class__
        elif other_inds.isssubset(self_inds):
            klass = self.__class__
        else: # neither one contains capabilities of the other
            klass = Affine
        a = klass()
        a._precond = self._precond
        a._vec12 = a._vector12(np.dot(self.as_affine(), other_aff))
        return a

    def __str__(self):
        string  = 'translation : %s\n' % str(self.translation)
        string += 'rotation    : %s\n' % str(self.rotation)
        string += 'scaling     : %s\n' % str(self.scaling)
        string += 'shearing    : %s' % str(self.shearing)
        return string

    def inv(self):
        """
        Return the inverse affine transform. 
        """
        a = self.__class__()
        a._precond = self._precond
        a._vec12 = a._vector12(spl.inv(self.as_affine()))
        return a


class Affine2D(Affine):
    param_inds = [0,1,5,6,7,11]


class Rigid(Affine):
    param_inds = range(6)

    def _vector12(self, aff):
        """
        Return a 12-sized vector of natural affine parameters:
        translation, rotation, log-scale, pre-rotation (to allow for
        shearing when combined with non-unitary scales).

        a better naming is
        vec12=[translation, post-rot, logscaling, pre-rot]
        """
        vec12 = np.zeros((12,))
        vec12[:3] = aff[:3,3]
        vec12[3:6] = rotation_mat2vec(aff[:3,:3])
        vec12[6:9] = 0.0
        return vec12

    def __str__(self):
        string  = 'translation : %s\n' % str(self.translation)
        string += 'rotation    : %s\n' % str(self.rotation)
        return string


class Rigid2D(Rigid):
    param_inds = [0,1,5]


class Similarity(Affine):
    param_inds = range(7)

    def _vector12(self, aff):
        """
        Return a 12-sized vector of natural affine parameters:
        translation, rotation, log-scale, pre-rotation (to allow for
        shearing when combined with non-unitary scales).

        a better naming is
        vec12=[translation, post-rot, logscaling, pre-rot]
        """
        vec12 = np.zeros((12,))
        vec12[:3] = aff[:3,3]
        ## A = s R ==> det A = (s)**3 ==> s = (det A)**(1/3)
        A = aff[:3,:3]
        s = spl.det(A)**(1/3.)
        vec12[3:6] = rotation_mat2vec(A/s)
        vec12[6:9] = np.log(np.maximum(s, TINY))
        return vec12

    def _set_param(self, p):
        p = np.asarray(p)
        self._vec12[range(9)] = (p[[0,1,2,3,4,5,6,6,6]] *
                                 self._precond[range(9)])

    param = property(Affine._get_param, _set_param)

    def __str__(self): 
        string  = 'translation : %s\n' % str(self.translation)
        string += 'rotation    : %s\n' % str(self.rotation)
        string += 'scaling     : %s\n' % str(self.scaling[0])
        return string


class Similarity2D(Similarity):
    param_inds = [0, 1, 5, 6]

    def _set_param(self, p):
        p = np.asarray(p)
        self._vec12[[0,1,5,6,7,8]] = (p[[0,1,2,3,3,3]] *
                                    self._precond[[0,1,5,6,7,8]])

    param = property(Similarity._get_param, _set_param)

