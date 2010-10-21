# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from nipy.externals.transforms3d.quaternions import mat2quat, quat2axangle
from ._registration import param_to_vector12, matrix44, affines, _affines

# Globals 
naffines = 3
id_rigid = affines.index('rigid')
id_similarity = affines.index('similarity')
id_affine = affines.index('affine')

# Defaults 
_radius = 100
_flag2d = False


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


def vector12(mat, subtype=id_affine):
    """
    Return a 12-sized vector of natural affine parameters:
    translation, rotation, log-scale, pre-rotation (to allow for
    shearing when combined with non-unitary scales).

    a better naming is
    vec12=[translation, post-rot, logscaling, pre-rot]

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


def inverse_affine(affine):
    return np.linalg.inv(affine)


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


class Affine(object): 

    def __init__(self, array=None, radius=_radius, flag2d=_flag2d):
        subtype = id_affine
        self._generic_init(array, radius, subtype, flag2d)
    
    def _generic_init(self, array, radius, subtype, flag2d): 
        if array == None: 
            self._vec12 = np.zeros(12)
        elif array.shape == (4,4):
            self._vec12 = vector12(array)
        elif array.size == 12: 
            self._vec12 = array.ravel()
        else: 
            raise ValueError('Invalid array')
        self._precond = preconditioner(radius)
        self._subtype = subtype
        self._flag2d = flag2d
        self._stamp = subtype + naffines*(not flag2d)

    def apply(self, xyz): 
        return apply_affine(self.as_affine(), xyz)

    def _get_param(self): 
        param = self._vec12/self._precond
        return param[_affines[self._stamp]]

    def _set_param(self, p): 
        self._vec12 = param_to_vector12(np.asarray(p), self._vec12, 
                                        self._precond, self._stamp)
        
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

    param = property(_get_param, _set_param)
    translation = property(_get_translation, _set_translation)
    rotation = property(_get_rotation, _set_rotation)
    scaling = property(_get_scaling, _set_scaling)
    shearing = property(_get_shearing, _set_shearing)
    precond = property(_get_precond)

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
        aff = self.as_affine()
        # Deliberately raise error in non-affine case for now
        other_aff = other.as_affine()
        return self.__class__(np.dot(aff, other_aff))

    def __str__(self): 
        string  = 'translation : %s\n' % str(self.translation)
        string += 'rotation    : %s\n' % str(self.rotation)
        string += 'scaling     : %s\n' % str(self.scaling)
        string += 'shearing    : %s' % str(self.shearing)
        return string

    def __mul__(self, other): 
        """
        Affine composition: T1oT2(x)
        """
        subtype = max(self._subtype, other._subtype)
        a = affine_classes[subtype]()
        a._precond = self._precond
        a._vec12 = vector12(np.dot(self.as_affine(), other.as_affine()), a._subtype)
        return a

    def inv(self):
        """
        Return the inverse affine transform. 
        """
        a = affine_classes[self._subtype]()
        a._precond = self._precond
        a._vec12 = vector12(np.linalg.inv(self.as_affine()), self._subtype)
        return a
        

class Rigid(Affine):

    def __init__(self, array=None, radius=_radius, flag2d=_flag2d):
        subtype = id_rigid
        self._generic_init(array, radius, subtype, flag2d)
        
    def __str__(self): 
        string  = 'translation : %s\n' % str(self.translation)
        string += 'rotation    : %s\n' % str(self.rotation)
        return string

class Similarity(Affine):

    def __init__(self, array=None, radius=_radius, flag2d=_flag2d):
        subtype = id_similarity
        self._generic_init(array, radius, subtype, flag2d)

    def __str__(self): 
        string  = 'translation : %s\n' % str(self.translation)
        string += 'rotation    : %s\n' % str(self.rotation)
        string += 'scaling     : %s\n' % str(self.scaling[0])
        return string
    

# List of 
affine_classes = [Rigid, Similarity, Affine]

