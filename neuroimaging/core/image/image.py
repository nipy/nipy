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

from neuroimaging.data_io.datasource import DataSource, splitzipext
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.reference.coordinate_system import CoordinateSystem
from neuroimaging.data_io.formats.format import getformats

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

    def _getaffine(self):
        if hasattr(self.grid, "affine"):
            return self.grid.affine
        raise AttributeError
    affine = property(_getaffine)

    def _getheader(self):
        if hasattr(self, '_header'):
            return self._header
        raise AttributeError
    header = property(_getheader)

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

        self._data = data
        self._grid = grid

    def _getshape(self):
        if hasattr(self._data, "shape"):
            return self._data.shape
        else:
            return self._data[:].shape
    shape = property(_getshape)

    def _getndim(self):
        if hasattr(self._data, "ndim"):
            return self._data.ndim
        else:
            return self._data[:].ndim
    ndim = property(_getndim)

    def _getgrid(self):
        return self._grid
    grid = property(_getgrid)

    # NOTE: Rename grid to spacemap?  There's been much discussion regarding
    # the appropriate name of this attr.  We should probably settle it and 
    # move on.

    def __getitem__(self, index):
        """Get a slice of image data.  Just like slicing a numpy array.

        Examples
        --------
        >>> from neuroimaging.core.image import image
        >>> from neuroimaging.testing import anatfile
        >>> img = image.load(anatfile)
        >>> zdim, ydim, xdim = img.shape
        >>> central_axial_slice = img[zdim/2, :, :]

        """

        if type(index) not in [type(()), type([])]:
            index = (index,)
        else:
            index = tuple(index)
        
        for i in index:
            if type(i) not in [type(1), type(slice(0,4,1))]:
                raise ValueError, 'when slicing images, index must be a list of integers or slices'
        data = self._data[index]
        grid = self.grid[index]
        return Image(data, grid)
    
    def __setitem__(self, slice_, data):
        """Set values of ``slice_`` to ``data``.
        """
        self._data[slice_] = data

    def __array__(self):
        """Return data in ndarray.  Called through numpy.array.
        
        Examples
        --------
        >>> import numpy as np
        >>> from neuroimaging.core.image import image
        >>> img = image.fromarray(np.zeros((21, 64, 64), dtype='int16'),
        ...                       ['zspace', 'yspace', 'xspace'])
        >>> imgarr = np.array(img)

        """

        return np.asarray(self._data[:])

## def _open_pynifti(url, datasource=DataSource(), mode="r", clobber=False):
##     """Open the image using PyNifti."""

##     """Look at the code from nifti1.Nifti1 for creating SamplingGrid

##     """

##     # Note: Nipy's Format code loads the data using a memmap.  Pynifti just
##     # loads it as an array.  But, np.allclose returns True when comparing
##     # the same file loaded through Pynifit and Nipy.
##     import nifti

##     from neuroimaging.core.reference.axis import space
##     import numpy as np

##     nimg = nifti.NiftiImage(url)

##     origin = nimg.getQOffset()
##     pixdim = nimg.getPixDims()
##     step = pixdim[0:3] # Note, getPixDims ignores pixdim[0], the qfac
##     dim = nimg.header.get('dim')  # Is there a method for getting this?
##     shape = tuple(dim[1:4])
##     ndim = dim[0]
##     if ndim == 3:
##         axisnames = space[::-1]  # Why do we reverse the list here?
##     else:
##         raise NotImplementedError, 'Not handling 4d images yet!'
    
##     grid = SamplingGrid.from_start_step(names=axisnames, 
##                                         shape=shape,
##                                         start=-np.array(origin),
##                                         step=step)
##     #raise NotImplementedError
##     return Image(nimg.asarray(), grid)

def _open(url, datasource=DataSource(), format=None, grid=None, mode="r",
          clobber=False, **keywords):
    """
    Create an `Image` from the given url/filename

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

    # remove any zip extensions
    url = splitzipext(url)[0]

    if not format:
        valid = getformats(url)
    else:
        valid = [format]
    errors = {}
    imgfmt = {}
    for format in valid:
        try:
            imgfmt = format(filename=url,
                          datasource=datasource, mode=mode,
                          grid=grid, clobber=clobber, **keywords)
        except Exception, exception:
            errors[format] = exception

    if imgfmt:
        tmpimg = Image(imgfmt, imgfmt.grid)
        tmpimg._header = imgfmt.header
        return tmpimg

    ## if file exists and clobber is False, error message is misleading
    # python test_image.py
    # <neuroimaging.core.reference.grid.SamplingGrid object at 0x1b6b310>
    # Traceback (most recent call last):
    #   File "test_image.py", line 12, in <module>
    #     image.save(img, "/home/cburns/data/avganat3.nii")
    #   File "/home/cburns/src/nipy-trunk/neuroimaging/core/image/image.py", line 102, in save
    #     format=format)
    #   File "/home/cburns/src/nipy-trunk/neuroimaging/core/image/image.py", line 77, in _open
    #     'Filename "%s" (or its header files) does not exist' % url
    # IOError: Filename "/home/cburns/data/avganat3.nii" (or its header files) does not exist

    ## same name, different file extension raises the same error

    ## problem: the exceptions raised are not IOErrors

    for format, exception in errors.items():
        if not exception.__class__  is IOError:
            raise NotImplementedError, 'no valid format found for URL %s.' \
                'The following errors were raised:\n%s' % \
                (url, "\n".join(["%s: %s\n%s" % \
                (str(format), str(msg.__class__), str(msg)) \
                for format, msg in errors.items()]))

    raise IOError, \
        'Filename "%s" (or its header files) does not exist' % url


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

def save(img, filename, datasource=DataSource(), clobber=False, format=None, **keywords):
    """Write the image to a file.

    Parameters
    ----------
    img : An `Image` object
    filename : string
        Should be a valid filename.
    datasource : A `DataSource` object
        A datasource to specify the location of the file.
    clobber : bool
        Should ``save`` overwrite an existing file.
    format : A `Format` object
        The image file format to save the
    keywords : Keyword arguments passed to `Format` initialization call.

    Returns
    -------
    image : An `Image` object

    See Also
    --------
    load : function for loading images
    fromarray : function for creating images from numpy arrays

    Examples
    --------
    
    # BUG:  image.save below will fail if keyword
    # "clobber=True" is left out. This is an IOError similar to
    # ones documented in the _open function above.
    
    >>> from numpy import allclose, array
    >>> from neuroimaging.core.image import image
    >>> from neuroimaging.testing import anatfile
    >>> img_orig = image.load(anatfile)
    >>> # For testing, we'll use a tempfile for saving.
    >>> # In 'real' work, you would save to a known directory.
    >>> from tempfile import mkstemp
    >>> fd, filename = mkstemp(suffix='.nii')
    >>> image.save(img_orig, filename, clobber=True)
    >>> img_copy = image.load(filename)
    >>> print allclose(array(img_orig)[:], array(img_copy)[:])
    True

    """

    # BUG:  If format is None, what's the default format?  Does it default
    # to nifti or look at the file extension?

    # Answer: looks at the file extension..

    outimage = _open(filename, mode='w', grid=img.grid,
                     clobber=clobber,
                     datasource=datasource,
                     format=format, **keywords)
    outimage[:] = np.array(img)[:]
    del(outimage)


def fromarray(data, names, grid=None):
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

def create_outfile(filename, grid, dtype=np.float32, clobber=False):
    """
    Create a zero-filled Image, saved to filename and
    reopened in 'r+' mode.
    """
    tmp = Image(np.zeros(grid.shape, dtype), grid)
    save(tmp, filename, clobber=clobber)
    del(tmp)
    return load(filename, mode='r+')

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


