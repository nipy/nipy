"""
Image interpolators using ndimage.
"""

import os

from scipy import ndimage
from neuroimaging import traits
import numpy as N

from neuroimaging.data_io.cache import cached

class ImageInterpolator(traits.HasTraits):

    order = traits.Int(1)

    def __init__(self, image, order=1, grid=None, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        if grid is None:
            self.grid = image.grid
        else:
            self.grid = grid
        self.image = image
        self.order = order
        self._prefilter()

    def _prefilter(self):
        if self.order > 1:
            data = ndimage.spline_filter(N.nan_to_num(self.image.readall()),
                                          self.order)
        else:
            data = N.nan_to_num(self.image.readall())

        if not hasattr(self, 'datafile'):
            self.datafile = file(cached(), 'rb+')
        else:
            self.datafile = file(self.datafile.name, 'rb+')
        
        data = N.nan_to_num(data.astype(N.float64))
        data.tofile(self.datafile)
        datashape = data.shape
        dtype = data.dtype

        del(data)
        self.datafile.close()

        self.datafile = file(self.datafile.name)
        self.data = N.memmap(self.datafile.name, dtype=dtype,
                             mode='r+', shape=datashape)

    def __del__(self):
        if hasattr(self, 'datafile'):
            self.datafile.close()
            try:
                os.remove(self.datafile.name)
            except:
                pass

    def __call__(self, points):
        return self.evaluate(points)

    def evaluate(self, points):
        points = N.array(points, N.float64)
        output_shape = points.shape[1:]
        points.shape = (points.shape[0], N.product(output_shape))
        voxels = self.grid.mapping.map.inverse()(points)
        V = ndimage.map_coordinates(self.data, 
                                     voxels,
                                     order=self.order,
                                     prefilter=False)
                                     
        V.shape = output_shape
        return V

    def resample(self, grid, mapping=None, **keywords):
        """
        Using an ImageInterpolator, resample an Image on the range
        of a grid, applying an optional mapping (taking
        keywords arguments **keywords) between the output
        coordinates of grid and self.image.grid.
        """

        points = grid.range()
        if mapping is not None:
            points = mapping(points, **keywords)
        return self.evaluate(points)
