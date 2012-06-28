# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import scipy.linalg as spl

from nibabel.affines import apply_affine

from ...externals.transforms3d.quaternions import mat2quat, quat2axangle
from .transform import Transform

# Globals
RADIUS = 100
MAX_ANGLE = 1e10 * 2 * np.pi
SMALL_ANGLE = 1e-30
MAX_DIST = 1e10
LOG_MAX_DIST = np.log(MAX_DIST)
TINY = float(np.finfo(np.double).tiny)


def threshold(x, th):
    return np.maximum(np.minimum(x, th), -th)


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

    To avoid numerical instabilities, an upper threshold is applied to
    the angle. It is chosen to be a multiple of 2*pi, hence the
    resulting rotation is then the identity matrix. This strategy warrants
    that the output matrix is a continuous function of the input vector.
    """
    theta = np.sqrt(np.sum(r ** 2))
    if theta > MAX_ANGLE:
        return np.eye(3)
    elif theta > SMALL_ANGLE:
        n = r / theta
        Sn = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        R = np.eye(3) + np.sin(theta) * Sn\
            + (1 - np.cos(theta)) * np.dot(Sn, Sn)
    else:
        Sr = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        theta2 = theta * theta
        R = np.eye(3) + (1 - theta2 / 6.) * Sr\
            + (.5 - theta2 / 24.) * np.dot(Sr, Sr)
    return R


def to_matrix44(t, dtype=np.double):
    """
    T = to_matrix44(t)

    t is a vector of affine transformation parameters with size at
    least 6.

    size < 6 ==> error
    size == 6 ==> t is interpreted as translation + rotation
    size == 7 ==> t is interpreted as translation + rotation +
    isotropic scaling
    7 < size < 12 ==> error
    size >= 12 ==> t is interpreted as translation + rotation +
    scaling + pre-rotation
    """
    size = t.size
    T = np.eye(4, dtype=dtype)
    R = rotation_vec2mat(t[3:6])
    if size == 6:
        T[0:3, 0:3] = R
    elif size == 7:
        T[0:3, 0:3] = t[6] * R
    else:
        S = np.diag(np.exp(threshold(t[6:9], LOG_MAX_DIST)))
        Q = rotation_vec2mat(t[9:12])
        # Beware: R*s*Q
        T[0:3, 0:3] = np.dot(R, np.dot(S, Q))
    T[0:3, 3] = threshold(t[0:3], MAX_DIST)
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
    (translation+rotation+scaling+pre-rotation) so that each element
    roughly represents a variation in mm.
    """
    rad = 1. / radius
    sca = 1. / radius
    return np.array([1, 1, 1, rad, rad, rad, sca, sca, sca, rad, rad, rad])


def inverse_affine(affine):
    return spl.inv(affine)


def subgrid_affine(affine, slices):
    steps = map(lambda x: max(x, 1), [s.step for s in slices])
    starts = map(lambda x: max(x, 0), [s.start for s in slices])
    t = np.diag(np.concatenate((steps, [1]), 1))
    t[0:3, 3] = starts
    return np.dot(affine, t)


class Affine(Transform):
    param_inds = range(12)

    def __init__(self, array=None, radius=RADIUS):
        self._direct = True
        self._precond = preconditioner(radius)
        if array == None:
            self._vec12 = np.zeros(12)
            return
        array = np.array(array)
        if array.size == 12:
            self._vec12 = array.ravel().copy()
        elif array.shape == (4, 4):
            self.from_matrix44(array)
        else:
            raise ValueError('Invalid array')

    def copy(self):
        new = self.__class__()
        new._direct = self._direct
        new._precond[:] = self._precond[:]
        new._vec12 = self._vec12.copy()
        return new

    def from_matrix44(self, aff):
        """
        Convert a 4x4 matrix describing an affine transform into a
        12-sized vector of natural affine parameters: translation,
        rotation, log-scale, pre-rotation (to allow for shearing when
        combined with non-unitary scales). In case the transform has a
        negative determinant, set the `_direct` attribute to False.
        """
        vec12 = np.zeros((12,))
        vec12[0:3] = aff[:3, 3]
        # Use SVD to find orthogonal and diagonal matrices such that
        # aff[0:3,0:3] == R*S*Q
        R, s, Q = spl.svd(aff[0:3, 0:3]) 
        if spl.det(R) < 0:
            R = -R
            Q = -Q
        r = rotation_mat2vec(R)
        if spl.det(Q) < 0:
            Q = -Q
            self._direct = False
        q = rotation_mat2vec(Q)
        vec12[3:6] = r
        vec12[6:9] = np.log(np.maximum(s, TINY))
        vec12[9:12] = q
        self._vec12 = vec12

    def apply(self, xyz):
        return apply_affine(self.as_affine(), xyz)

    def _get_param(self):
        param = self._vec12 / self._precond
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

    def _get_pre_rotation(self):
        return self._vec12[9:12]

    def _set_pre_rotation(self, x):
        self._vec12[9:12] = x

    def _get_direct(self):
        return self._direct

    def _get_precond(self):
        return self._precond

    translation = property(_get_translation, _set_translation)
    rotation = property(_get_rotation, _set_rotation)
    scaling = property(_get_scaling, _set_scaling)
    pre_rotation = property(_get_pre_rotation, _set_pre_rotation)
    is_direct = property(_get_direct)
    precond = property(_get_precond)
    param = property(_get_param, _set_param)

    def as_affine(self, dtype='double'):
        T = to_matrix44(self._vec12, dtype=dtype)
        if not self._direct:
            T[:3, :3] *= -1
        return T

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
        # If other is not an Affine, use either its left compose
        # method, if available, or the generic compose method
        if not hasattr(other, 'as_affine'):
            if hasattr(other, 'left_compose'):
                return other.left_compose(self)
            else:
                return Transform(self.apply).compose(other)

        # Affine case: choose more capable of input types as output
        # type
        other_aff = other.as_affine()
        self_inds = set(self.param_inds)
        other_inds = set(other.param_inds)
        if self_inds.issubset(other_inds):
            klass = other.__class__
        elif other_inds.isssubset(self_inds):
            klass = self.__class__
        else:  # neither one contains capabilities of the other
            klass = Affine
        a = klass()
        a._precond[:] = self._precond[:]
        a.from_matrix44(np.dot(self.as_affine(), other_aff))
        return a

    def __str__(self):
        string = 'translation : %s\n' % str(self.translation)
        string += 'rotation    : %s\n' % str(self.rotation)
        string += 'scaling     : %s\n' % str(self.scaling)
        string += 'pre-rotation: %s' % str(self.pre_rotation)
        return string

    def inv(self):
        """
        Return the inverse affine transform.
        """
        a = self.__class__()
        a._precond[:] = self._precond[:]
        a.from_matrix44(spl.inv(self.as_affine()))
        return a


class Affine2D(Affine):
    param_inds = [0, 1, 5, 6, 7, 11]


class Rigid(Affine):
    param_inds = range(6)

    def from_matrix44(self, aff):
        """
        Convert a 4x4 matrix describing a rigid transform into a
        12-sized vector of natural affine parameters: translation,
        rotation, log-scale, pre-rotation (to allow for pre-rotation
        when combined with non-unitary scales). In case the transform
        has a negative determinant, set the `_direct` attribute to
        False.
        """
        vec12 = np.zeros((12,))
        vec12[:3] = aff[:3, 3]
        R = aff[:3, :3]
        if spl.det(R) < 0:
            R = -R
            self._direct = False
        vec12[3:6] = rotation_mat2vec(R)
        vec12[6:9] = 0.0
        self._vec12 = vec12

    def __str__(self):
        string = 'translation : %s\n' % str(self.translation)
        string += 'rotation    : %s\n' % str(self.rotation)
        return string


class Rigid2D(Rigid):
    param_inds = [0, 1, 5]


class Similarity(Affine):
    param_inds = range(7)

    def from_matrix44(self, aff):
        """
        Convert a 4x4 matrix describing a similarity transform into a
        12-sized vector of natural affine parameters: translation,
        rotation, log-scale, pre-rotation (to allow for pre-rotation
        when combined with non-unitary scales). In case the transform
        has a negative determinant, set the `_direct` attribute to
        False.
        """
        vec12 = np.zeros((12,))
        vec12[:3] = aff[:3, 3]
        ## A = s R ==> det A = (s)**3 ==> s = (det A)**(1/3)
        A = aff[:3, :3]
        detA = spl.det(A)
        s = np.maximum(np.abs(detA) ** (1 / 3.), TINY)
        if detA < 0:
            A = -A
            self._direct = False
        vec12[3:6] = rotation_mat2vec(A / s)
        vec12[6:9] = np.log(s)
        self._vec12 = vec12

    def _set_param(self, p):
        p = np.asarray(p)
        self._vec12[range(9)] =\
            (p[[0, 1, 2, 3, 4, 5, 6, 6, 6]] * self._precond[range(9)])

    param = property(Affine._get_param, _set_param)

    def __str__(self):
        string = 'translation : %s\n' % str(self.translation)
        string += 'rotation    : %s\n' % str(self.rotation)
        string += 'scaling     : %s\n' % str(self.scaling[0])
        return string


class Similarity2D(Similarity):
    param_inds = [0, 1, 5, 6]

    def _set_param(self, p):
        p = np.asarray(p)
        self._vec12[[0, 1, 5, 6, 7, 8]] =\
            (p[[0, 1, 2, 3, 3, 3]] * self._precond[[0, 1, 5, 6, 7, 8]])

    param = property(Similarity._get_param, _set_param)


affine_transforms = {'affine': Affine,
                     'affine2d': Affine2D,
                     'similarity': Similarity,
                     'similarity2d': Similarity2D,
                     'rigid': Rigid,
                     'rigid2d': Rigid2D}
