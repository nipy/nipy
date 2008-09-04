"""The image module provides basic functions for working with images in nipy.
Functions are provided to load, save and create image objects, along with
iterators to easily slice through volumes.

    load : load an image from a file or url

    save : save an image to a file

    fromarray : create an image from a numpy array

Examples
--------
See documentation for load and save functions for 'working' examples.

"""

__docformat__ = 'restructuredtext'
__all__ = ['load', 'save', 'fromarray']

import numpy as np

from neuroimaging.io.datasource import DataSource, splitzipext
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.reference.mapping import Affine

from neuroimaging.io.pyniftiio import PyNiftiIO, orientation_to_names

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
    >>> img = image.load(anatfile)

    >>> import numpy as np
    >>> img = image.fromarray(np.zeros((21, 64, 64), dtype='int16'),
    ...                       ['zspace', 'yspace', 'xspace'])

    """

    def __init__(self, data, grid):
        """Create an `Image` object from a numpy array and a `Grid` object.
        
        Images should be created through the module functions load and
        fromarray.

        Parameters
        ----------
        data : A numpy.ndarray
        grid : A `SamplingGrid` Object
        
        See Also
        --------
        load : load `Image` from a file
        save : save `Image` to a file
        fromarray : create an `Image` from a numpy array

        """

        if data is None or grid is None:
            raise ValueError, 'expecting an array and SamplingGrid instance'

        # self._data is an array-like object.  It must implement a subset of
        # array methods  (Need to specify these, for now implied in pyniftio)
        self._data = data
        self._grid = grid

    def _getshape(self):
        return self._data.shape
    shape = property(_getshape, doc="Shape of data array")

    def _getndim(self):
        return self._data.ndim
    ndim = property(_getndim, doc="Number of data dimensions")

    def _getgrid(self):
        return self._grid
    grid = property(_getgrid,
                    doc="Coordinate mapping from input coords to output coords")

    def _getaffine(self):
        if hasattr(self.grid, "affine"):
            return self.grid.affine
        raise AttributeError, 'Nonlinear transform does not have an affine.'
    affine = property(_getaffine, doc="Affine transformation is one exists")

    def _getheader(self):
        # data loaded from a file should have a header
        if hasattr(self._data, 'header'):
            return self._data.header
        raise AttributeError, 'Image created from arrays do not have headers.'
    header = property(_getheader, doc="Image header if loaded from disk")

    def __getitem__(self, index):
        """Slicing an image returns a new image."""
        data = self._data[index]
        grid = self.grid[index]
        # BUG: If it's a zero-dimension array we should return a numpy scalar
        # like np.int32(data[index])
        # Need to figure out elegant way to handle this
        return Image(data, grid)

    def __setitem__(self, index, value):
        """Setting values of an image, set values in the data array."""
        self._data[index] = value

    def __array__(self):
        """Return data as a numpy array."""
        return np.asarray(self._data)

def _open(url, datasource=DataSource(), format=None, grid=None, mode="r",
          clobber=False, **keywords):
    """Create an `Image` from the given url/filename

    Parameters
    ----------
    url : ``string``
        a url or filename
    datasource : `DataSource`
        The datasource to be used for caching
    format : `Format`
        The file format to use. If ``None`` then all possible formats will
        be tried.
    grid : `reference.grid.SamplingGrid`
        The sampling grid for the file
    mode : ``string``
        The mode ot open the file in ('r', 'w', etc)

    Returns
    -------
    image : A new `Image` object created from the url.

    Notes
    -----
    Raises IOError : If the specified format, or those tried by default
        all raise IOErrors.
    Raises NotImplementedError : If the specified format, or those tried by
        default are unable to open the file, an exception is raised.

    The raising of an exception can be misleading. If for example, a
    bad url is given, it will appear as if that file's format has not
    been implemented.
    
    """

    try:
        ioimg = PyNiftiIO(url, mode)
        if grid is None:
            grid = grid_from_affine(ioimg.affine, ioimg.orientation,
                                    ioimg.shape)
        # Build nipy image from array-like object and sampling grid
        img = Image(ioimg, grid)
        return img
    except IOError:
        raise IOError, 'Unable to open file %s' % url
        
def load(url, datasource=DataSource(), format=None, mode='r', **keywords):
    """Load an image from the given url.

    Load an image from the file specified by ``url`` and ``datasource``.

    Parameters
    ----------
    url : string
        Should resolve to a complete filename path, possibly with the provided
        datasource.
    datasource : A `DataSource` object
        A datasource for the image to load.
    format : A `Format` object
        The file format to use when opening the image file.  If ``None``, the
        default, all supported formats are tried.
    mode : Either 'r' or 'r+'
    keywords : Keyword arguments passed to `Format` initialization call.

    Returns
    -------
    image : An `Image` object
        If successful, a new `Image` object is returned.

    See Also
    --------
    save : function for saving images
    fromarray : function for creating images from numpy arrays

    Notes
    -----
    The raising of an exception can be misleading. If for example, a bad url 
    is given, it will appear as if that file's format has not been implemented.

    Examples
    --------

    >>> from neuroimaging.core.image import image
    >>> from neuroimaging.testing import funcfile
    >>> img = image.load(funcfile)
    >>> img.shape
    (20, 2, 20, 20)

    """

    # BUG: Should DataSource here be a Repository?  So the Repository
    # would be a 'base url' and the url would be the filename.
    # Fix when porting code to numpy-trunk that now includes DataSource.
    # and update documentation above.

    if mode not in ['r', 'r+']:
        raise ValueError, 'image opening mode must be either "r" or "r+"'
    return _open(url, datasource=datasource, format=format, mode=mode, **keywords)

def save(img, filename, datasource=DataSource()):
    """Write the image to a file.

    Parameters
    ----------
    img : An `Image` object
    filename : string
        Should be a valid filename.
    datasource : A `DataSource` object
        A datasource to specify the location of the file.
        NOTE: This is currently ignored!

    Returns
    -------
    image : An `Image` object

    See Also
    --------
    load : function for loading images
    fromarray : function for creating images from numpy arrays

    Notes
    -----
    Filetype is determined by the file extension in 'filename'.  Currently the
    following filetypes are supported:
        Nifti single file : ['.nii', '.nii.gz']
        Nifti file pair : ['.hdr', '.hdr.gz']
        Analyze file pair : ['.img', 'img.gz']
        
    """

    data = np.asarray(img)
    outimage = _open(data, grid=img.grid, mode='w')
    outimage._data.save(filename)
    return outimage
    
def fromarray(data, names=['zspace', 'yspace', 'xspace'], grid=None):
    """Create an image from a numpy array.

    Parameters
    ----------
    data : numpy array
        A numpy array of three dimensions.
    names : a list of axis names
    grid : A `SamplingGrid`
        If not specified, a uniform sampling grid is created.

    Returns
    -------
    image : An `Image` object

    See Also
    --------
    load : function for loading images
    save : function for saving images

    """

    ndim = len(data.shape)
    if not grid:
        grid = SamplingGrid.from_start_step(names,
                                            (0,)*ndim,
                                            (1,)*ndim,
                                            data.shape)

    return Image(data, grid)

def merge_images(filename, images, cls=Image, clobber=False,
                 axis='time'):
    """
    Create a new file based image by combining a series of images together.

    Parameters
    ----------
    filename : ``string``
        The filename to write the new image as
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
    grid = im0.grid.replicate(n, axis)
    data = np.empty(shape=grid.shape)
    for i, image in enumerate(images):
        data[i] = np.asarray(image)[:]
    return Image(data, grid)

def zeros(grid):
    """
    Return an Image of zeros with a given grid.
    """
    return Image(np.zeros(grid.shape), grid)


def grid_from_affine(affine, orientation, shape):
    """Generate a SamplingGrid from an affine transform."""

    """
    spaces = ['vector','time','zspace','yspace','xspace']
    space = tuple(spaces[-ndim:])
    shape = tuple(img.header['dim'][1:ndim+1])
    grid = SamplingGrid.from_affine(Affine(affine),space,shape)
    return grid        
    """
    names = []
    for ornt in orientation:
        names.append(orientation_to_names.get(ornt))
    names = names[::-1]
    if len(shape) == 4:
        names = ['time'] + names
    elif len(shape) == 5:
        names = ['vector', 'time'] + names
    affobj = Affine(affine)
    grid = SamplingGrid.from_affine(affobj, names, shape)
    return grid
