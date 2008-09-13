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
    TODO
    """

    def __init__(self, image, order=1, comap=None):
        """
        :Parameters:
            image : TODO
                TODO
            order : ``int``
                TODO
            comap : TODO
                TODO        
        """
        if comap is None:
            self.comap = image.comap
        else:
            self.comap = comap
        self.image = image
        self.order = order
        self._prefilter()

    def _prefilter(self):
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

    def __call__(self, points):
        """
        :Parameters:
            points : TODO
                TODO

        :Returns: TODO
        """
        return self.evaluate(points)

    def evaluate(self, points):
        """
        :Parameters:
            points : TODO
                TODO

        :Returns: TODO
        """
        points = np.array(points, np.float64)
        output_shape = points.shape[1:]
        points.shape = (points.shape[0], np.product(output_shape))
        voxels = self.comap.mapping.inverse()(points)
        V = ndimage.map_coordinates(self.data, 
                                     voxels,
                                     order=self.order,
                                     prefilter=False)
                                     
        V.shape = output_shape
        return V

    def resample(self, comap, mapping=None, **keywords):
        """
        Using an ImageInterpolator, resample an Image on the range
        of a comap, applying an optional mapping (taking
        keyword arguments ``keywords``) between the output
        coordinates of comap and self.image.comap.

        :Parameters:
            comap : TODO
                TODO
            mapping : TODO
                TODO
            keywords : ``dict``
                TODO

        :Returns: TODO        
        """

        points = comap.range()
        if mapping is not None:
            points = mapping(points, **keywords)
        return self.evaluate(points)
