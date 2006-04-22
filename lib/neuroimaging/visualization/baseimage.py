""" BaseImage class - wrapper for Image class to test changes to Image interface
"""

from neuroimaging.image import Image
from neuroimaging.data import FileSystem
from neuroimaging.reference.grid import IdentityGrid

class BaseImage(object):
    
    def __init__(self, pathname, datasource=FileSystem(), reader=Image):
        self.pathname = pathname
        self.datasource = datasource
        self._image = reader(pathname, datasource=datasource)

    def _get_array(self):
        return self._image.readall()

    array = property(_get_array)

    def _get_grid(self):
        """ Gets the grid from the image

        If there is no grid, create a default
        """
        if not hasattr(self._image, 'grid'):
            self._image.grid = IdentityGrid(self.array.shape)
        
        return self._image.grid

    grid = property(_get_grid)

    def _get_shape(self):
        return self.grid.shape

    shape = property(_get_shape)
    
    def _get_ndim(self):
        return len(self.grid.shape)

    shape = property(_get_ndim)

