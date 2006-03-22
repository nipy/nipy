"""
Image interpolators using nd_image.
"""

import scipy.nd_image as nd_image
import enthought.traits as traits
import tempfile, os
from neuroimaging.reference import grid
from neuroimaging.cache import cached
import numpy as N

class ImageInterpolator(traits.HasTraits):

    order = traits.Int(1)

    def __init__(self, image, order=1, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.image = image
        self.prefilter()

    def prefilter(self):
        if self.order > 1:
            data = nd_image.spline_filter(N.nan_to_num(self.image.readall()),
                                          self.order)
        else:
            data = N.nan_to_num(self.image.readall())

        if not hasattr(self, 'datafile'):
            self.datafile = file(cached(), 'rb+')
        else:
            self.datafile = file(self.datafile.name, 'rb+')
        
        data = N.nan_to_num(data.astype(N.Float))
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
            os.remove(self.datafile.name)

    def __call__(self, points):
        return self.evaluate(points)

    def evaluate(self, points):
        if isinstance(points, grid.SamplingGrid):
            points = points.range() # optimize this for affine grids!
        points = N.array(points, N.Float)
        output_shape = points.shape[1:]
        points.shape = (points.shape[0], N.product(output_shape))
        voxels = self.image.grid.mapping.map(points, inverse=True)
        V = nd_image.map_coordinates(self.data, 
                                     voxels,
                                     order=self.order,
                                     prefilter=False)
                                     
        V.shape = output_shape
        return V
