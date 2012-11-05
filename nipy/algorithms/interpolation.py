# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Image interpolators using ndimage.
"""

import os
import tempfile

import numpy as np

from scipy import ndimage

class ImageInterpolator(object):
    """ Interpolate Image instance at arbitrary points in world space
    
    The resampling is done with scipy.ndimage.
    """
    def __init__(self, image, order=3):
        """
        Parameters
        ----------
        image : Image
           Image to be interpolated
        order : int, optional
           order of spline interpolation as used in scipy.ndimage.
           Default is 3.
        """
        self.image = image
        self.order = order
        self._datafile = None
        self._buildknots()

    def _buildknots(self):
        if self.order > 1:
            data = ndimage.spline_filter(
                np.nan_to_num(self.image.get_data()),
                self.order)
        else:
            data = np.nan_to_num(self.image.get_data())
        if self._datafile is None:
            _, fname = tempfile.mkstemp()
            self._datafile = open(fname, mode='wb')
        else:
            self._datafile = open(self._datafile.name, 'wb')
        data = np.nan_to_num(data.astype(np.float64))
        data.tofile(self._datafile)
        datashape = data.shape
        dtype = data.dtype
        del(data)
        self._datafile.close()
        self._datafile = open(self._datafile.name)
        self.data = np.memmap(self._datafile.name, dtype=dtype,
                              mode='r+', shape=datashape)

    def __del__(self):
        if self._datafile:
            self._datafile.close()
            try:
                os.remove(self._datafile.name)
            except:
                pass

    def evaluate(self, points):
        """ Resample image at points in world space
        
        Parameters
        ----------
        points : array
           values in self.image.coordmap.output_coords.  Each row is a
	   point. 

        Returns
        -------
        V : ndarray
           interpolator of self.image evaluated at points
        """
        points = np.array(points, np.float64)
        output_shape = points.shape[1:]
        points.shape = (points.shape[0], np.product(output_shape))
        cmapi = self.image.coordmap.inverse()
        voxels = cmapi(points.T).T
        V = ndimage.map_coordinates(self.data, 
                                     voxels,
                                     order=self.order,
                                     prefilter=False)
        # ndimage.map_coordinates returns a flat array,
        # it needs to be reshaped to the original shape
        V.shape = output_shape
        return V
