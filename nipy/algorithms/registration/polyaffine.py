from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from .transform import Transform
from .affine import apply_affine
from ._registration import _apply_polyaffine

TINY_SIGMA = 1e-200


class PolyAffine(Transform):

    def __init__(self, centers, affines, sigma, glob_affine=None):
        """
        centers: N times 3 array

        We are given a set of affine transforms T_i with centers x_i,
        all in homogeneous coordinates. The polyaffine transform is
        defined, up to a right composition with a global affine, as:

        T(x) = sum_i w_i(x) T_i x

        where w_i(x) = g(x-x_i)/Z(x) are normalized Gaussian weights
        that sum up to one for every x.
        """

        # Format input arguments
        self.centers = np.asarray(centers, dtype='double', order='C')
        self.sigma = np.zeros(3)
        self.sigma[:] = np.maximum(TINY_SIGMA, sigma)
        if hasattr(affines[0], 'as_affine'):
            affines = np.array([a.as_affine() for a in affines])
        else:
            affines = np.asarray(affines)
        if hasattr(glob_affine, 'as_affine'):
            self.glob_affine = glob_affine.as_affine()
        else:
            self.glob_affine = glob_affine

        # Cache a (N, 12) matrix containing the affines coefficients,
        # should be C-contiguous double.
        self._affines = np.zeros((len(self.centers), 12))
        self._affines[:] = np.reshape(affines[:, 0:3, :],
                                      (len(self.centers), 12))

    def affine(self, i):
        aff = np.eye(4)
        aff[0:3, :] = self._affines[i].reshape(3, 4)
        return aff

    def affines(self):
        return [self.affine(i) for i in range(len(self.centers))]

    def apply(self, xyz):
        """
        xyz is an (N, 3) array
        """
        # txyz should be double C-contiguous for the the cython
        # routine _apply_polyaffine
        if self.glob_affine is None:
            txyz = np.array(xyz, copy=True, dtype='double', order='C')
        else:
            txyz = apply_affine(self.glob_affine, xyz)
        _apply_polyaffine(txyz, self.centers, self._affines, self.sigma)
        return txyz

    def compose(self, other):
        """
        Compose this transform onto another

        Parameters
        ----------
        other : Transform
            transform that we compose onto

        Returns
        -------
        composed_transform : Transform
            a transform implementing the composition of self on `other`
        """
        # If other is not an Affine, use the generic compose method
        if not hasattr(other, 'as_affine'):
            return Transform(self.apply).compose(other)

        # Affine case: the result is a polyaffine transform with same
        # local affines
        if self.glob_affine is None:
            glob_affine = other.as_affine()
        else:
            glob_affine = np.dot(self.glob_affine, other.as_affine())

        return self.__class__(self.centers, self.affines(), self.sigma,
                              glob_affine=glob_affine)

    def left_compose(self, other):

        # If other is not an Affine, use the generic compose method
        if not hasattr(other, 'as_affine'):
            return Transform(other.apply).compose(self)

        # Affine case: the result is a polyaffine transform with same
        # global affine
        other_affine = other.as_affine()
        affines = [np.dot(other_affine, self.affine(i)) \
                       for i in range(len(self.centers))]
        return self.__class__(self.centers, affines, self.sigma,
                              glob_affine=self.glob_affine)
