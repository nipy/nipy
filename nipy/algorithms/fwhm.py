# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module provides classes and definitions for using full width at half
maximum (FWHM) to be used in conjunction with Gaussian Random Field Theory
to determine resolution elements (resels).

A resolution element (resel) is defined as a block of pixels of the same
size as the FWHM of the smoothed image.

There are two methods implemented to estimate (3d, or volumewise) FWHM
based on a 4d Image:

    fastFHWM: used if the entire 4d Image is available
    iterFWHM: used when 4d Image is being filled in by slices of residuals
    
"""

__docformat__ = 'restructuredtext'

import numpy as np
from numpy.linalg import det

from nipy.core.api import Image

from .utils.matrices import pos_recipr

class Resels(object):
    """The Resels class.
    """
    def __init__(self, coordmap, normalized=False, fwhm=None, resels=None,
                 mask=None, clobber=False, D=3):
        """ Initialize resels class

        Parameters
        ----------
        coordmap : ``CoordinateMap``
                CoordinateMap over which fwhm and resels are to be estimated.
                Used in fwhm/resel conversion.
        fwhm : ``Image``
            Optional Image of FWHM. Used to convert
            FWHM Image to resels if FWHM is not being estimated.
        resels : ``Image``
            Optional Image of resels. Used to
            compute resels within a mask, for instance, if
            FWHM has already been estimated.
        mask : ``Image``
            Mask over which to integrate resels.
        clobber : ``bool``
            Clobber output FWHM and resel images?
        D : ``int``
            Can be 2 or 3, the dimension of the final volume.
        """
        self.fwhm = fwhm
        self.resels = resels
        self.mask = mask
        self.clobber = clobber
        self.coordmap = coordmap
        self.D = D
        self.normalized = normalized
        _transform = self.coordmap.affine
        self.wedge = np.power(np.fabs(det(_transform)), 1./self.D)

    def integrate(self, mask=None):
        """ Integrate resels within `mask` (or use self.mask)

        Parameters
        ----------
        mask : ``Image``
                Optional mask over which to integrate (add) resels.

        Returns
        -------
        total_resels :
            the resels contained in the mask
        FWHM : float
            an estimate of FWHM based on the average resel per voxel
        nvoxel: int
            the number of voxels in the mask
        """
        _resels = self.resels[:]
        if mask is not None:
            _mask = mask
        else:
            _mask = self.mask
        if _mask is not None:
            _mask = _mask[:].astype(np.int32)
            nvoxel = _mask.sum()
        else:
            _mask = 1.
            nvoxel = _resels.size
        _resels = (_resels * _mask).sum()
        _fwhm = self.resel2fwhm(_resels / nvoxel)
        return _resels, _fwhm, nvoxel

    def resel2fwhm(self, resels):
        """ Convert resels as `resels` to isotropic FWHM

        Parameters
        ----------
        resels : float
            Convert a resel value to an equivalent isotropic FWHM based on
            step sizes in self.coordmap.

        Returns
        -------
        fwhm : float
        """
        return np.sqrt(4*np.log(2.)) * self.wedge * pos_recipr(np.power(resels, 1./self.D))

    def fwhm2resel(self, fwhm):
        """ Convert FWHM `fwhm` to equivalent reseels per voxel

        Parameters
        ----------
        fwhm : float
            Convert an FWHM value to an equivalent resels per voxel based on
            step sizes in self.coordmap.


        Returns
        -------
        resels : float
        """
        return pos_recipr(np.power(fwhm / np.sqrt(4*np.log(2)) * self.wedge, self.D))

    def __iter__(self):
        """ Return iterator

        Returns
        -------
        itor : iterator
            self
        """
        if not self.fwhm:
            im = Image(np.zeros(self.resid.shape), coordmap=self.coordmap)
        else:
            im = \
              Image(self.fwhm, clobber=self.clobber, mode='w', coordmap=self.coordmap)
        self.fwhm = im

        if not self.resels:
            im = Image(np.zeros(self.resid.shape), coordmap=self.coordmap)
        else:
            im = \
              Image(self.resels, clobber=self.clobber, mode='w', coordmap=self.coordmap)
        self.resels = im

        return self


class ReselImage(Resels):

    def __init__(self, resels=None, fwhm=None, **keywords):
        """ Initialize resel image

        Parameters
        ----------
        resels : `core.api.Image`
            Image of resel per voxel values.
        fwhm : `core.api.Image`
            Image of FWHM values.
        keywords : ``dict``
            Passed as keywords arguments to `core.api.Image`
        """
        if not resels and not fwhm:
            raise ValueError, 'need either a resels image or an FWHM image'

        if fwhm is not None:
            fwhm = Image(fwhm, **keywords)
            Resels.__init__(self, fwhm, resels=resels, fwhm=fwhm)

        if resels is not None:
            resels = Image(resels, **keywords)
            Resels.__init__(self, resels, resels=resels, fwhm=fwhm)

        if not self.fwhm:
            self.fwhm = Image(self.resel2fwhm(self.resels[:]),
                              coordmap=self.coordmap, **keywords)

        if not self.resels:
            self.resels = Image(self.fwhm2resel(self.fwhm[:]),
                                coordmap=self.coordmap, **keywords)

    def __iter__(self):
        """ Return iterator

        Returns
        -------
        itor : iterator
            ``self``
        """
        return self


def _calc_detlam(xx, yy, zz, yx, zx, zy):
    """
    Calculate determinant of symmetric 3x3 matrix

    [[xx,yx,xz],
     [yx,yy,zy],
     [zx,zy,zz]]
     """
    
    return zz * (yy*xx - yx**2) - \
           zy * (zy*xx - zx*yx) + \
           zx * (zy*yx - zx*yy)
