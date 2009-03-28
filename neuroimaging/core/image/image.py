"""
This module defines the Image class, as well as two functions that create Image instances.

fromarray : create an Image instance from an ndarray

merge_images : create an Image by merging a sequence of Image instance

"""
import numpy as np

from neuroimaging.core.reference.coordinate_map import reorder_input, reorder_output, Affine
from neuroimaging.core.reference.coordinate_map import product as cmap_product
from neuroimaging.core.reference.coordinate_system import CoordinateSystem
from neuroimaging.core.reference.array_coords import ArrayCoordMap

__docformat__ = 'restructuredtext'
__all__ = ['fromarray']

class Image(object):
    """
    The `Image` class provides the core object type used in nipy. An `Image`
    represents a volumetric brain image and provides means for manipulating
    the image data.  Most functions in the image module operate on `Image`
    objects.

    Notes
    -----
    Images should be created through the module functions load and fromarray.

    Examples
    --------

    >>> from neuroimaging.core.image import image
    >>> from neuroimaging.testing import anatfile
    >>> from neuroimaging.io.api import load_image
    >>> img = load_image(anatfile)

    >>> import numpy as np
    >>> img = image.fromarray(np.zeros((21, 64, 64), dtype='int16'),
    ...                       'kji', 'zxy')

    """

    # Dictionary to store docs for attributes that are properties.  We
    # want these docs to conform with our documentation standard, but
    # they need to be passed into the property function.  Defining
    # them separately allows us to do this without a lot of clutter
    # int he property line.
    _doc = {}
    
    def __init__(self, data, coordmap):
        """Create an `Image` object from array and ``CoordinateMap`` object.
        
        Images should be created through the module functions load and
        fromarray.

        Parameters
        ----------
        data : A numpy.ndarray
        coordmap : A `CoordinateMap` Object
        
        See Also
        --------
        load : load `Image` from a file
        save : save `Image` to a file
        fromarray : create an `Image` from a numpy array

        """

        if data is None or coordmap is None:
            raise ValueError('expecting an array and CoordinateMap instance')

        # This ensures two things
        # i) each axis in coordmap.input_coords has a length and
        # ii) the shapes are consistent

        # self._data is an array-like object.  It must implement a subset of
        # array methods  (Need to specify these, for now implied in pyniftio)
        self._data = data
        self._coordmap = coordmap

    def _getshape(self):
        return self._data.shape
    shape = property(_getshape, doc="Shape of data array")

    def _getndim(self):
        return self._data.ndim
    ndim = property(_getndim, doc="Number of data dimensions")

    def _getcoordmap(self):
        return self._coordmap
    coordmap = property(_getcoordmap,
                    doc="Coordinate mapping from input coords to output coords")

    def _getaffine(self):
        if hasattr(self.coordmap, "affine"):
            return self.coordmap.affine
        raise AttributeError, 'Nonlinear transform does not have an affine.'
    affine = property(_getaffine, doc="Affine transformation is one exists")

    def _getheader(self):
        # data loaded from a file should have a header
        if hasattr(self._data, 'header'):
            return self._data.header
        raise AttributeError, 'Image created from arrays do not have headers.'
    def _setheader(self, header):
        if hasattr(self._data, 'header'):
            self._data.header = header
        else:
            raise AttributeError, \
                  'Image created from arrays do not have headers.'
    _doc['header'] = \
    """The file header dictionary for this image.  In order to update
    the header, you must first make a copy of the header, set the
    values you wish to change, then set the image header to the
    updated header.

    Example
    -------

    hdr = img.header
    hdr['slice_duration'] = 0.200
    hdr['descrip'] = 'My image registered with MNI152.'
    img.header = hdr
    
    """
    header = property(_getheader, _setheader, doc=_doc['header'])

    def __getitem__(self, index):
        """Slicing an image returns a new image."""
        data = self._data[index]
        g = ArrayCoordMap(self.coordmap, self._data.shape)[index]
        coordmap = g.coordmap
        # BUG: If it's a zero-dimension array we should return a numpy scalar
        # like np.int32(data[index])
        # Need to figure out elegant way to handle this
        return Image(data, coordmap)

    def __setitem__(self, index, value):
        """Setting values of an image, set values in the data array."""
        self._data[index] = value

    def __array__(self):
        """Return data as a numpy array."""
        return np.asarray(self._data)

def fromarray(data, innames, outnames, coordmap=None):
    """Create an image from a numpy array.

    Parameters
    ----------
    data : numpy array
        A numpy array of three dimensions.
    names : a list of axis names
    coordmap : A `CoordinateMap`
        If not specified, an identity coordinate map is created.

    Returns
    -------
    image : An `Image` object

    See Also
    --------
    load : function for loading images
    save : function for saving images

    """

    ndim = len(data.shape)
    if not coordmap:
        coordmap = Affine.from_start_step(innames,
                                          outnames,
                                          (0.,)*ndim,
                                          (1.,)*ndim)
                                          
    return Image(data, coordmap)

def merge_images(images, cls=Image, clobber=False,
                 axis='merge'):
    """
    Create a new file based image by combining a series of images together.
    The resulting CoordinateMap are essentially copies of images[0].coordmap

    Parameters
    ----------
    images : [`Image`]
        The list of images to be merged
    cls : ``class``
        The class of image to create
    clobber : ``bool``
        Overwrite the file if it already exists
    axis : ``string``
        Name of the concatenated axis.
        
    Returns
    -------
    ``cls``
    
    """
    
    n = len(images)
    im0 = images[0]
    coordmap = cmap_product(Affine(np.identity(2), 
                                   CoordinateSystem([axis]), 
                                   CoordinateSystem([axis])),
                            im0.coordmap)
    data = np.empty(shape=(n,) + im0.shape)
    for i, image in enumerate(images):
        data[i] = np.asarray(image)[:]
    return Image(data, coordmap)

