# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Linear filter(s).  For the moment, only a Gaussian smoothing filter
"""
from __future__ import absolute_import

import gc

import numpy as np
import numpy.fft as fft
import numpy.linalg as npl

from nipy.utils import seq_prod
from nipy.core.api import Image, AffineTransform
from nipy.core.reference.coordinate_map import product

class LinearFilter(object):
    '''
    A class to implement some FFT smoothers for Image objects.
    By default, this does a Gaussian kernel smooth. More choices
    would be better!
    '''

    normalization = 'l1sum'

    def __init__(self, coordmap, shape, fwhm=6.0, scale=1.0, location=0.0,
                 cov=None):
        """
        Parameters
        ----------
        coordmap : ``CoordinateMap``
        shape : sequence
        fwhm : float, optional
           fwhm for Gaussian kernel, default is 6.0
        scale : float, optional
           scaling to apply to data after smooth, default 1.0
        location : float
           offset to apply to data after smooth and scaling, default 0
        cov : None or array, optional
           Covariance matrix
        """
        self.coordmap = coordmap
        self.bshape = shape
        self.fwhm = fwhm
        self.scale = scale
        self.location = location
        self.cov = cov
        self._setup_kernel()

    def _setup_kernel(self):
        if not isinstance(self.coordmap, AffineTransform):
            raise ValueError('for FFT smoothing, we need a '
                             'regular (affine) coordmap')
        # voxel indices of array implied by shape
        voxels = np.indices(self.bshape).astype(np.float64)
        # coordinates of physical center.  XXX - why the 'floor' here?
        vox_center = np.floor((np.array(self.bshape) - 1) / 2.0)
        phys_center = self.coordmap(vox_center)
        # reshape to (N coordinates, -1).  We appear to need to assign
        # to shape instead of doing a reshape, in order to avoid memory
        # copies
        voxels.shape = (voxels.shape[0], seq_prod(voxels.shape[1:]))
        # physical coordinates relative to center
        X = (self.coordmap(voxels.T) - phys_center).T
        X.shape = (self.coordmap.ndims[1],) + tuple(self.bshape)
        # compute kernel from these positions
        kernel = self(X, axis=0)
        kernel = _crop(kernel)
        self.norms = {'l2':np.sqrt((kernel**2).sum()),
                      'l1':np.fabs(kernel).sum(),
                      'l1sum':kernel.sum()}
        self._kernel = kernel
        self.shape = (np.ceil(
            (np.asarray(self.bshape) + np.asarray(kernel.shape)) / 2)
            * 2 + 2).astype(np.intp)
        self.fkernel = np.zeros(self.shape)
        slices = [slice(0, kernel.shape[i]) for i in range(len(kernel.shape))]
        self.fkernel[slices] = kernel
        self.fkernel = fft.rfftn(self.fkernel)
        return kernel

    def _normsq(self, X, axis=-1):
        """
        Compute the (periodic, i.e. on a torus) squared distance needed for
        FFT smoothing. Assumes coordinate system is linear.

        Parameters
        ----------
        X : array
           array of points
        axis : int, optional
           axis containing coordinates. Default -1
        """
        # copy X
        _X = np.array(X)
        # roll coordinate axis to front
        _X = np.rollaxis(_X, axis)
        # convert coordinates to FWHM units
        if self.fwhm is not 1.0:
            f = fwhm2sigma(self.fwhm)
            if f.shape == ():
                f = np.ones(len(self.bshape)) * f
            for i in range(len(self.bshape)):
                _X[i] /= f[i]
        # whiten?
        if self.cov is not None:
            _chol = npl.cholesky(self.cov)
            _X = np.dot(npl.inv(_chol), _X)
        # compute squared distance
        D2 = np.sum(_X**2, axis=0)
        return D2

    def __call__(self, X, axis=-1):
        ''' Compute kernel from points

        Parameters
        ----------
        X : array
           array of points
        axis : int, optional
           axis containing coordinates.  Default -1
        '''
        _normsq = self._normsq(X, axis) / 2.
        t = np.less_equal(_normsq, 15)
        return np.exp(-np.minimum(_normsq, 15)) * t

    def smooth(self, inimage, clean=False, is_fft=False):
        """ Apply smoothing to `inimage`

        Parameters
        ----------
        inimage : ``Image``
           The image to be smoothed.  Should be 3D.
        clean : bool, optional
           Should we call ``nan_to_num`` on the data before smoothing?
        is_fft : bool, optional
           Has the data already been fft'd?

        Returns
        -------
        s_image : `Image`
           New image, with smoothing applied
        """
        if inimage.ndim == 4:
            # we need to generalize which axis to iterate over.  By
            # default it should probably be the last.
            raise NotImplementedError('Smoothing volumes in a 4D series '
                                      'is broken, pending a rethink')
            _out = np.zeros(inimage.shape)
            # iterate over the first (0) axis - this is confusing - see
            # above
            nslice = inimage.shape[0]
        elif inimage.ndim == 3:
            nslice = 1
        else:
            raise NotImplementedError('expecting either 3 or 4-d image')
        in_data = inimage.get_data()
        for _slice in range(nslice):
            if in_data.ndim == 4:
                data = in_data[_slice]
            elif in_data.ndim == 3:
                data = in_data[:]
            if clean:
                data = np.nan_to_num(data)
            if not is_fft:
                data = self._presmooth(data)
            data *= self.fkernel
            data = fft.irfftn(data) / self.norms[self.normalization]
            gc.collect()
            _dslice = [slice(0, self.bshape[i], 1) for i in range(3)]
            if self.scale != 1:
                data = self.scale * data[_dslice]
            if self.location != 0.0:
                data += self.location
            gc.collect()
            # Write out data 
            if in_data.ndim == 4:
                _out[_slice] = data
            else:
                _out = data
            _slice += 1
        gc.collect()
        _out = _out[[slice(self._kernel.shape[i] // 2,
                           self.bshape[i] + self._kernel.shape[i] // 2)
                     for i in range(len(self.bshape))]]
        if inimage.ndim == 3:
            return Image(_out, coordmap=self.coordmap)
        else:
            # This does not work as written.  See above
            concat_affine = AffineTransform.identity('concat')
            return Image(_out, coordmap=product(self.coordmap, concat_affine))

    def _presmooth(self, indata):
        slices = [slice(0, self.bshape[i], 1) for i in range(len(self.shape))]
        _buffer = np.zeros(self.shape)
        _buffer[slices] = indata
        return fft.rfftn(_buffer)


def fwhm2sigma(fwhm):
    """ Convert a FWHM value to sigma in a Gaussian kernel.

    Parameters
    ----------
    fwhm : array-like
       FWHM value or values

    Returns
    -------
    sigma : array or float
       sigma values corresponding to `fwhm` values

    Examples
    --------
    >>> sigma = fwhm2sigma(6)
    >>> sigmae = fwhm2sigma([6, 7, 8])
    >>> sigma == sigmae[0]
    True
    """
    fwhm = np.asarray(fwhm)
    return fwhm / np.sqrt(8 * np.log(2))


def sigma2fwhm(sigma):
    """ Convert a sigma in a Gaussian kernel to a FWHM value

    Parameters
    ----------
    sigma : array-like
       sigma value or values

    Returns
    -------
    fwhm : array or float
       fwhm values corresponding to `sigma` values

    Examples
    --------
    >>> fwhm = sigma2fwhm(3)
    >>> fwhms = sigma2fwhm([3, 4, 5])
    >>> fwhm == fwhms[0]
    True
    """
    sigma = np.asarray(sigma)
    return sigma * np.sqrt(8 * np.log(2))


def _crop(X, tol=1.0e-10):
    """
    Find a bounding box for support of fabs(X) > tol and returned
    crop region.
    """
    aX = np.fabs(X)
    n = len(X.shape)
    I = np.indices(X.shape)[:, np.greater(aX, tol)]
    if I.shape[1] > 0:
        m = [I[i].min() for i in range(n)]
        M = [I[i].max() for i in range(n)]
        slices = [slice(m[i], M[i]+1, 1) for i in range(n)]
        return X[slices]
    else:
        return np.zeros((1,)*n)

