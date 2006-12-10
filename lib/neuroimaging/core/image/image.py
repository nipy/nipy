"""
The core Image class.
"""

import types

import numpy as N

from neuroimaging import flatten
from neuroimaging.data_io import DataSource, splitzipext
from neuroimaging.data_io.formats import getformats, Format
from neuroimaging.core.image.base_image import ArrayImage

from neuroimaging.core.reference.iterators import SliceIterator

class Image(object):
    """
    The Image class provides the core object type used in nipy. An Image
    represents a volumetric brain image and provides means for manipulating and
    reading and writing this data to file.
    """

    @staticmethod
    def fromurl(url, datasource=DataSource(), format=None, grid=None, mode="r",
                **keywords):
        """
        Create an L{Image} from the given url/filename

        @param url: a url or filename
        @type url: C{string}
        @param datasource: The datasource to be used for caching
        @type datasource: L{DataSource}
        @param format: The file format to use. If C{None} then all possible
            formats will be tried.
        @type format: L{Format}
        @param grid: The sampling grid for the file
        @type grid: L{SamplingGrid}
        @param mode: The mode to open the file in ('r', 'w', etc)
        @type mode: C{string}

        @raise NotImplementedError: If the specified format, or those tried by
            default are unable to open the file, an exception is raised.

        @note: The raising of an exception can be misleading. If for example, a
            bad url is given, it will appear as if that file's format has not
            been implemented.

        @return: A new L{Image} created from C{url}
        @rtype: L{Image}
        """
        # remove any zip extensions
        url = splitzipext(url)[0]
            
        if not format:
            valid = getformats(url)
        else:
            valid = [format]
        errors = {}
        for format in valid:
            try:
                return format(filename=url,
                              datasource=datasource, mode=mode,
                              grid=grid, **keywords)
            except Exception, e:
                errors[format] = e

        raise NotImplementedError, 'no valid reader found for URL %s\n%s' % \
              (url, \
              "\n".join(["%s: %s\n%s" % (str(format), str(msg.__class__), str(msg)) for format, msg in errors.items()]))

    def __init__(self, image, datasource=DataSource(), grid=None, **keywords):
        '''
        Create an Image (volumetric image) object from either a file, an
        existing L{Image} object, or an array.

        @param image: This can be either an L{Image}, string or an array.
        '''

        # from existing Image
        if isinstance(image, Image):
            self._source = image._source

        # from existing Format instance
        elif isinstance(image, Format):
            self._source = image

        # from array
        elif isinstance(image, N.ndarray) or isinstance(image, N.core.memmap):
            self._source = ArrayImage(image, grid=grid)

        # from filename or url
        elif type(image) == types.StringType:
            self._source = \
              self.fromurl(image, datasource, grid=grid, **keywords)
        else:
            raise ValueError(
          "Image input must be a string, array, or another image.")

        # Find spatial grid -- this is the one that will be used generally
        self.grid = self._source.grid
        self.shape = list(self.grid.shape)
        self.ndim = len(self.shape)

        # Attach memory-mapped array or array as buffer attr
        self._source.data.shape = self.shape
        self.buffer = self._source.data


    def __getitem__(self, slice_):
        return self._source[slice_]


    def __setitem__(self, slice_, data):
        self._source[slice_] = data


    def __iter__(self):
        """ Images cannot be used directly as iterators. """
        raise NotImplementedError


    def toarray(self, clean=True, **keywords):
        """
       Return a Image instance that has an ArrayImage as its _source attribute.

        >>> from numpy import *
        >>> from BrainSTAT import *
        >>> test = Image(testfile('anat+orig.HEAD'))
        >>> _test = test.toarray()
        >>> print _test.source.data.shape
        (124, 256, 256)
        >>> test = Image(testfile('test_fmri.img'))
        >>> _test = test.toarray(slice=(2,), grid=test.grid)
        >>> print _test.shape
        (13, 128, 128)
        """
        data = self.readall()
        if clean and \
               data.dtype.type in N.sctypes['float'] + N.sctypes['complex']: 
            data = N.nan_to_num(data)
            
        return Image(data, grid=self.grid, **keywords)


    def tofile(self, filename, clobber=False,
               dtype=None, **keywords):
        """        
        Write the image to a file. Returns a new Image object
        of the newly written file.

        @param filename: The name of the file to write to
        @type filename: C{string}
        @param clobber: Should we overwrite an existing file?
        @type clobber: C{bool}
        
        """
        dtype = dtype or self._source.dtype
        outimage = Image(filename, mode='w', grid=self.grid,
                         clobber=clobber,
                         dtype=dtype,
                         **keywords)

        tmp = self.toarray(**keywords)
        outimage[:] = tmp[:]
        return outimage


    def readall(self, clean=False): 
        """
        Read an entire Image object, returning a numpy, not another instance of
        Image. By default, it does not read 4d images. Missing values are
        filled in with 0
        """
        value = self[self.grid.allslice()]
        if clean: 
            value = N.nan_to_num(value)
        return value


    def slice_iterator(self, mode='r', axis=0):
        ''' Return slice iterator for this image '''
        return SliceIterator(self, mode=mode, axis=axis)

    def from_slice_iterator(self, other, axis=0):
        it = iter(SliceIterator(self, mode='w', axis=axis))
        for s in other:
            it.next().set(s)

    def iterate(self, iterator):
        iterator.set_img(self)
        return iterator

    def from_iterator(self, other, iterator):
        iterator.mode = 'w'
        iterator.set_img(self)
        iter(iterator)
        for s in other:
            iterator.next().set(s)

class ImageSequenceIterator(object):
    """
    Take a sequence of images, and an optional grid (which defaults to
    imgs[0].grid) and create an iterator whose next method returns array with
    shapes (len(imgs),) + self.imgs[0].next().shape Very useful for voxel-based
    methods, i.e. regression, one-sample t.
    """
    def __init__(self, imgs, grid=None):
        self.imgs = imgs
        if grid is None:
            self.grid = self.imgs[0].grid
        else:
            self.grid = grid
        iter(self)

    def __iter__(self): 
        """ Return self as an iterator. """
        self.iters = [img.slice_iterator() for img in self.imgs]
        return self

    def next(self):
        """ Return the next iterator value. """
        val = [it.next() for it in self.iters]
        return N.array(val, N.float64)

