"""
The core Image class.
"""

__docformat__ = 'restructuredtext'

import numpy as N

from neuroimaging.data_io import DataSource, splitzipext
from neuroimaging.data_io.formats import getformats, Format
from neuroimaging.core.image.base_image import ArrayImage

from neuroimaging.core.reference.iterators import SliceIterator

class Image(object):
    """
    The `Image` class provides the core object type used in nipy. An `Image`
    represents a volumetric brain image and provides means for manipulating and
    reading and writing this data to file.
    """

    @staticmethod
    def fromurl(url, datasource=DataSource(), format=None, grid=None, mode="r",
                **keywords):
        """
        Create an `Image` from the given url/filename

        :Parameters:
            `url` : string
                a url or filename
            `datasource` : `DataSource`
                The datasource to be used for caching
            `format` : `Format`
                The file format to use. If ``None`` then all possible formats will
                be tried.
            `grid` : `reference.grid.SamplingGrid`
                The sampling grid for the file
            `mode` : string
                The mode ot open the file in ('r', 'w', etc)

        :Raises IOError: If the specified format, or those tried by default
            all raise IOErrors.
        :Raises NotImplementedError: If the specified format, or those tried by
            default are unable to open the file, an exception is raised.

        :Returns:
            `Image` : A new `Image` created from url

        Notes
        -----
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
        for format in valid:
            try:
                return format(filename=url,
                              datasource=datasource, mode=mode,
                              grid=grid, **keywords)
            except Exception, exception:
                errors[format] = exception

        for format, exception in errors.items():
            if not exception.__class__  is IOError:
                raise NotImplementedError, 'no valid format found for URL %s.' \
                      'The following errors were raised:\n%s' % \
                      (url, \
                       "\n".join(["%s: %s\n%s" % \
                                  (str(format), str(msg.__class__), str(msg)) \
                                  for format, msg in errors.items()]))

        raise IOError, \
              'Filename "%s" (or its header files) does not exist' % url

    def __init__(self, image, datasource=DataSource(), grid=None, **keywords):
        '''
        Create an `Image` (volumetric image) object from either a file, an
        existing `Image` object, or an array.

        :Parameters:
            `image` : `Image` or ``string`` or ``array``
                The object to create this Image from. If an `Image` or ``array``
                are provided, their data is used. If a string is given it is treated
                as either a filename or url.
        '''

        # from existing Image
        if isinstance(image, Image):
            self._source = image._source

        # from existing Format instance
        elif isinstance(image, Format):
            self._source = image

        # from array
        elif isinstance(image, (N.ndarray, N.core.memmap)):
            self._source = ArrayImage(image, grid=grid)

        # from filename or url
        elif isinstance(image, str):
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
        """ `Image`\ s cannot be used directly as iterators.

        :Raises NotImplementedError:
        """
        raise NotImplementedError


    def toarray(self, clean=True, **keywords):
        """
        Return a `Image` instance that has an `ArrayImage` as its _source attribute.

        Example
        -------
        
        >>> from numpy import *
        >>> from neuroimaging.core.image.image import Image
        >>> from neuroimaging.utils.tests.data import repository
        >>> test = Image('anat+orig.HEAD', datasource=repository)
        >>> _test = test.toarray()
        >>> print _test._source.data.shape
        (124, 256, 256)
        >>> test = Image('test_fmri.img', datasource=repository)
        >>> _test = test.toarray()
        >>> print _test.shape
        [120, 13, 128, 128]
        
        """
        data = self.readall()
        if clean and \
               data.dtype.type in N.sctypes['float'] + N.sctypes['complex']: 
            data = N.nan_to_num(data)
            
        return Image(data, grid=self.grid, **keywords)


    def tofile(self, filename, clobber=False,
               dtype=None, **keywords):
        """        
        Write the image to a file. Returns a new `Image` object
        of the newly written file.

        :Parameters:
            `filename` : string
                The name of the file to write to
            `clobber` : bool
                Should we overwrite an existing file?

        :Returns: `Image`
        """
        dtype = dtype or self._source.dtype
        outimage = Image(filename, mode='w', grid=self.grid,
                         clobber=clobber,
                         dtype=dtype,
                         **keywords)

        tmp = self.toarray(**keywords)
        outimage[:] = tmp[:]
        return outimage


    def asfile(self):
        """ Return image filename corresponding to `Image` object data

        :Returns: ``string``
        """
        filename = self._source.asfile()
        if isinstance(self._source, ArrayImage):
            self.tofile(filename, clobber=True)
        return filename

    def readall(self, clean=False): 
        """
        Read an entire `Image` object, returning a numpy array, not another
        instance of `Image`. By default, it does not read 4d images. Missing
        values are filled in with 0
        """
        value = self[self.grid.allslice()]
        if clean: 
            value = N.nan_to_num(value)
        return value


    def slice_iterator(self, mode='r', axis=0):
        """ Return slice iterator for this image

        :Parameters:
            `axis` : int or [int]
                The index of the axis (or axes) to be iterated over. If a list
                is supplied the axes are iterated over slowest to fastest.
            `mode` : string
                The mode to run the iterator in.
                'r' - read-only (default)
                'w' - read-write

        :Returns:
            `SliceIterator`
        """
        return SliceIterator(self, mode=mode, axis=axis)

    def from_slice_iterator(self, other, axis=0):
        """
        Take an existing `SliceIterator` and use it to set the values
        in this image.

        :Parameters:
            `other` : `SliceIterator`
                The iterator from which to take the values
            `axis` : int or [int]
                The axis to iterator over for this image.
        """
        iterator = iter(SliceIterator(self, mode='w', axis=axis))
        for slice_ in other:
            iterator.next().set(slice_)

    def iterate(self, iterator):
        """
        Use the given iterator to iterate over this image.

        :Parameters:
            `iterator` : `reference.iterators.Iterator`
                The iterator to use.

        :Returns:
            An iterator which can be used to iterate over this image.
        """
        iterator.set_img(self)
        return iter(iterator)

    def from_iterator(self, other, iterator):
        """
        Set the values of this image, taking them from one iterator and using
        another to do the iteration over itself.

        :Parameters:
            `other` : `reference.iterators.Iterator`
                The iterator from which to take the values
            `iterator` : `reference.iterators.Iterator`
                The iterator to use to iterate over self.
        """
        iterator.mode = 'w'
        iterator.set_img(self)
        iter(iterator)
        for slice_ in other:
            iterator.next().set(slice_)

class ImageSequenceIterator(object):
    """
    Take a sequence of `Image`s, and an optional grid (which defaults to
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
        self.iters = None
        iter(self)

    def __iter__(self): 
        """ Return self as an iterator. """
        self.iters = [img.slice_iterator() for img in self.imgs]
        return self

    def next(self):
        """ Return the next iterator value. """
        val = [it.next() for it in self.iters]
        return N.array(val, N.float64)

