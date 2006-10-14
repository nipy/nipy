"""
The core Image class.
"""

import types

import numpy as N

from neuroimaging import flatten
from neuroimaging.data_io import DataSource, splitzipext
from neuroimaging.data_io.formats import getformats, Format
from neuroimaging.core.image.base_image import ArrayImage


class Image(object):
    """
    The Image class provides the core object type used in nipy. An Image
    represents a volumetric brain image and provides means for manipulating and
    reading and writing this data to file.
    """

    @staticmethod
    def fromurl(url, datasource=DataSource(), format=None, grid=None, mode="r",
                clobber=False, **keywords):
        """
        Create an Image from the given url/filename
        """
        
        # remove any zip extensions
        url = splitzipext(url)[0]
            
        if not format:
            valid = getformats(url)
        else:
            valid = [format]
        for format in valid:
            try:
                return format(filename=url,
                              datasource=datasource, mode=mode, clobber=clobber,
                              grid=grid, **keywords)
            except Exception, e:
            #    print e
                pass

        raise NotImplementedError, 'no valid reader found for URL %s' % url

    def __init__(self, image, datasource=DataSource(), grid=None, **keywords):
        '''
        Create a Image (volumetric image) object from either a file, an
        existing Image object, or an array.
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
        self.buffer = self._source.data


    def __getitem__(self, slice_):
        return self._source[slice_]

    def __setitem__(self, slice_, data):
        self._source[slice_] = data


    def __iter__(self):
        """ Create an iterator over an image based on its grid's iterator."""
        iter(self.grid)
        return self


    def compress(self, where, axis=None):        
        """
        Call the compress method on the underlying data array
        """
        return self.buffer.compress(where, axis=axis)


    def put(self, indices, data):
        """
        Call the put method on the underlying data array
        """
        return self.buffer.put(indices, data)


    def next(self, value=None, data=None):
        """
        The value argument here is used when, for instance one wants to
        iterate over one image with a ParcelIterator and write out data
        to this image without explicitly setting this image's grid to
        the original image's grid, i.e. to just take the value the
        original image's iterator returns and use it here.
        """
        if value is None:
            value = self.grid.next()
        itertype = self.grid.get_iter_param("itertype")

        if data is None:
            if itertype is 'slice':
                result = N.squeeze(self[value.slice])
            elif itertype is 'parcel':
                flatten(value.where)
                result = self.compress(value.where)
            elif itertype == 'slice/parcel':
                result = self[value.slice].compress(value.where)
            return result
        else:
            if itertype is 'slice':
                self[value.slice] = data
            elif itertype in ('parcel', "slice/parcel"):
                self.put(N.nonzero(value.where.flatten()), data)


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
               sctype=None, **keywords):
        """        
        Write the image to a file. Returns a new Image object
        of the newly written file.
        """
        sctype = sctype or self._source.sctype
        outimage = Image(filename, mode='w', grid=self.grid,
                         clobber=clobber,
                         sctype=sctype,
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
            self.grid = iter(self.imgs[0].grid)
        else:
            self.grid = iter(grid)

    def __iter__(self): 
        return self

    def next(self, value=None):
        if value is None:
            value = self.grid.next()
        val = [img.next(value=value) for img in self.imgs]
        return N.array(val, N.float64)

