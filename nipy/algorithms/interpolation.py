# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Image interpolators using ndimage.
"""
from __future__ import absolute_import

import tempfile

from distutils.version import LooseVersion as LV

import numpy as np

import scipy
from scipy import ndimage

from ..fixes.scipy.ndimage import map_coordinates
from ..utils import seq_prod


# Earlier versions of Scipy don't have mode for spline_filter
SCIPY_VERSION = LV(scipy.__version__)
SPLINE_FILTER_HAS_MODE = SCIPY_VERSION >= LV('1.2')
# Fixes in interpolation in scipy >= 1.6 force pre-padding
# in knot calculations.
SPLINE_FILTER_NEEDS_PAD = SCIPY_VERSION >= LV('1.6')


class ImageInterpolator(object):
    """ Interpolate Image instance at arbitrary points in world space

    The resampling is done with ``scipy.ndimage``.
    """

    # Padding for prefilter calculation in 'nearest' and 'grid-constant' mode.
    # See: https://github.com/scipy/scipy/issues/13600
    n_prepad_if_needed = 12 if SPLINE_FILTER_NEEDS_PAD else 0

    def __init__(self, image, order=3, mode='constant', cval=0.0):
        """
        Parameters
        ----------
        image : Image
           Image to be interpolated.
        order : int, optional
           order of spline interpolation as used in ``scipy.ndimage``.
           Default is 3.
        mode : str, optional
           Points outside the boundaries of the input are filled according to
           the given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
           is 'constant'.
        cval : scalar, optional
           Value used for points outside the boundaries of the input if
           mode='constant'. Default is 0.0.
        """
        # order and mode are read-only to allow pre-calculation of spline
        # filters.
        self.image = image
        self._order = order
        self._mode = mode
        self.cval = cval
        self._datafile = None
        self._n_prepad = 0  # Non-zero for 'nearest' and 'grid-constant'
        self._buildknots()

    @property
    def mode(self):
        """ Mode is read-only
        """
        return self._mode

    @property
    def order(self):
        """ Order is read-only
        """
        return self._order

    def _buildknots(self):
        data = np.nan_to_num(self.image.get_data()).astype(np.float64)
        if self.order > 1:
            if self.mode in ('nearest', 'grid-constant'):
                # See: https://github.com/scipy/scipy/issues/13600
                self._n_prepad = self.n_prepad_if_needed
                if self._n_prepad != 0:
                    data = np.pad(data, self._n_prepad, mode='edge')
            kwargs = dict(order=self.order)
            if SPLINE_FILTER_HAS_MODE:
                kwargs['mode'] = self.mode
            data = ndimage.spline_filter(data, **kwargs)
        self._datafile = tempfile.TemporaryFile()
        data.tofile(self._datafile)
        self._data = np.memmap(self._datafile,
                               dtype=data.dtype,
                               mode='r+',
                               shape=data.shape)
        del(data)

    def evaluate(self, points):
        """ Resample image at points in world space

        Parameters
        ----------
        points : array
           values in self.image.coordmap.output_coords.  Each row is a point.

        Returns
        -------
        V : ndarray
           interpolator of self.image evaluated at points
        """
        points = np.array(points, np.float64)
        output_shape = points.shape[1:]
        points.shape = (points.shape[0], seq_prod(output_shape))
        cmapi = self.image.coordmap.inverse()
        voxels = cmapi(points.T).T + self._n_prepad
        V = map_coordinates(self._data,
                            voxels,
                            order=self.order,
                            mode=self.mode,
                            cval=self.cval,
                            prefilter=self.order < 2)
        # ndimage.map_coordinates returns a flat array,
        # it needs to be reshaped to the original shape
        V.shape = output_shape
        return V
