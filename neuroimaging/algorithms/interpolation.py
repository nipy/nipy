"""
Image interpolators using ndimage.
"""

__docformat__ = 'restructuredtext'

import os
import numpy as np

from scipy import ndimage
from neuroimaging.io.api import Cache

class ImageInterpolator(object):
    """
    A class that enables interpolation of an Image instance
    at arbitrary points in the Image's world space (image.coordmap.output_coords).

    The resampling is done with scipy.ndimage.
    """

    def __init__(self, image, order=3):
        """
        :Parameters:
            image : Image
                Image to be interpolated
            order : ``int``
                order of spline interpolation as used in scipy.ndimage
                
        """
        self.image = image
        self.order = order
        self._buildknots()

    def _buildknots(self):
        if self.order > 1:
            data = ndimage.spline_filter(np.nan_to_num(np.asarray(self.image)),
                                          self.order)
        else:
            data = np.nan_to_num(np.asarray(self.image))

        if not hasattr(self, 'datafile'):
            self.datafile = file(Cache().tempfile(), mode='wb')
        else:
            self.datafile = file(self.datafile.name, 'wb')
        
        data = np.nan_to_num(data.astype(np.float64))
        data.tofile(self.datafile)
        datashape = data.shape
        dtype = data.dtype

        del(data)
        self.datafile.close()

        self.datafile = file(self.datafile.name)
        self.data = np.memmap(self.datafile.name, dtype=dtype,
                             mode='r+', shape=datashape)

    def __del__(self):
        if hasattr(self, 'datafile'):
            self.datafile.close()
            try:
                os.remove(self.datafile.name)
            except:
                pass

    def evaluate(self, points):
        """
        :Parameters:
            points : values in self.image.coordmap.output_coords 

        :Returns: 
            V: ndarray
               interpolator of self.image evaluated at points

        """
        points = np.array(points, np.float64)
        output_shape = points.shape[1:]
        points.shape = (points.shape[0], np.product(output_shape))
        voxels = self.image.coordmap.inverse(points.T).T
        V = ndimage.map_coordinates(self.data, 
                                     voxels,
                                     order=self.order,
                                     prefilter=False)
                                     
        # ndimage.map_coordinates returns a flat array,
        # it needs to be reshaped to the original shape
        
        V.shape = output_shape
        return V

