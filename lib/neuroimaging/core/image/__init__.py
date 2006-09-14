"""
The Image class provides the interface which should be used
by users at the application level. It is build onto of a
BaseImage object (self._source) which handles the actual
representation of the data. A base image provides a grid,
a data type and the data itself, while the main Image class
builds on top of these.

A BaseImage object can be created from an ndarray (ArrayImage)
or from a file (Formats). 


TODO: The Formats class does not currently subclass from BaseImage,
or even provide the correct interface, so this needs to be fixed
to fall in line with the class structure shown here.

  Application Level
----------------------
        Image
          |
          o
          |
      BaseImage
          |
          |
      ------------
      |          |
   Formats   ArrayImage
      |
   Binary   
      |
  ------------------
  |        |       |
Nifti   Analyze  ECAT
"""



import types, os

from neuroimaging import traits
import numpy as N

from neuroimaging import flatten
from neuroimaging.data_io import DataSource
from neuroimaging.data_io.formats import getformats, Format
from neuroimaging.core.reference.grid import SamplingGrid

class BaseImage(object):
    """
    This class define a minimal interface which different types of images
    should implement. This interface is used by the Image class, which is
    the class which should be used by applications.
    """

    def __init__(self, data, grid, sctype):
        assert(data.shape == grid.shape)
        self.grid = grid
        self.data = data
        self.sctype = sctype
        
    def __getitem__(self, item):
        return self.data[item]
        
    def __setitem__(self, item, value):
        self.data[item] = value

class ArrayImage (BaseImage):
    """A simple class to mimic an image file from an array."""
    def __init__(self, data, grid=None):
        """
        Create an ArrayImage instance from an array,
        by default assumed to be 3d.

        >>> from numpy import *
        >>> from neuroimaging.core.image import Image
        >>> z = Image.ArrayImage(zeros((10,20,20)))
        >>> print z.ndim
        3
        """
        grid = grid and grid or SamplingGrid.identity(self.shape)
        sctype = data.dtype.type
        BaseImage.__init__(self, data, grid, sctype)



class Image(object):

    @staticmethod
    def fromurl(url, datasource=DataSource(), format=None, grid=None, mode="r", clobber=False,
      **keywords):
        zipexts = (".gz",".bz2")
        base, ext = os.path.splitext(url.strip())
        if ext in zipexts: url = base
        if not format: valid = getformats(url)
        else: valid = [format]
        for format in valid:
            try:
                return format(filename=url,
                               datasource=datasource, mode=mode, clobber=clobber, grid=grid, **keywords)
            except Exception, e:
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
            self._source = self.fromurl(image, datasource, grid=grid, **keywords)

        else:
            raise ValueError(
          "Image input must be a string, array, or another image.")
        # Find spatial grid -- this is the one that will be used generally
        self.grid = self._source.grid
        self.shape = list(self.grid.shape)
        self.ndim = len(self.shape)

        # When possible, attach memory-mapped array or array as buffer attr
        if hasattr(self._source, 'memmap'):
            self.buffer = self._source.memmap
        else:
            self.buffer = self._source.data

        self.postread = lambda x:x
        self.fill = 0.0


    def __getitem__(self, slice): return self._source[slice]
    def __setitem__(self, slice, data): self._source[slice] = data


    def __iter__(self):
        "Create an iterator over an image based on its grid's iterator."
        iter(self.grid)
        if self.grid.get_iter_param("itertype") in ["parcel", "slice/parcel"]:
            self.buffer.shape = N.product(self.buffer.shape)
        return self


    def compress(self, where, axis=0):
        if hasattr(self, 'buffer'):
            return self.buffer.compress(where, axis=axis)
        else: raise ValueError, 'no buffer: compress not supported'


    def put(self, data, indices):
        if hasattr(self, 'buffer'):
            return self.buffer.put(data, indices)
        else: raise ValueError, 'no buffer: put not supported'


    def next(self, value=None, data=None):
        """
        The value argument here is used when, for instance one wants to
        iterate over one image with a ParcelIterator and write out data
        to this image without explicitly setting this image's grid to
        the original image's grid, i.e. to just take the value the
        original image's iterator returns and use it here.
        """
        if value is None:
            self.itervalue = value = self.grid.next()
        itertype = self.grid.get_iter_param("itertype")

        if data is None:
            if itertype is 'slice':
                result = N.squeeze(self[value.slice])
                #result = self[value.slice]
            elif itertype is 'parcel':
                flatten(value.where)
                self.label = value.label
                result = self.compress(value.where, axis=0)
            elif itertype == 'slice/parcel':
                result = self[value.slice].compress(value.where)
            return self.postread(result)
        else:
            if itertype is 'slice':
                self[value.slice] = data
            elif itertype in ('parcel', "slice/parcel"):
                self.put(data, N.nonzero(value.where.flatten()))


    def getvoxel(self, voxel):
        if len(voxel) != self.ndim:
            raise ValueError, 'expecting a voxel coordinate'
        return self[voxel]


    def toarray(self, clean=True, **keywords):
        """
        Return a Image instance that has an ArrayPipe as its image attribute.

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
        if clean and data.dtype.type in N.sctypes['float'] + N.sctypes['complex']: 
            data = N.nan_to_num(data)
            
        return Image(self.postread(data), grid=self.grid, **keywords)


    def tofile(self, filename, array=True, clobber=False,
               sctype=None, **keywords):
        sctype = sctype or self._source.sctype
        outimage = Image(filename, mode='w', grid=self.grid,
                         clobber=clobber,
                         sctype=sctype,
                         **keywords)
        if array:
            tmp = self.toarray(**keywords)
            outimage._source[:] = tmp._source.data
        else:
            tmp = iter(self)
            outimage = iter(outimage)
            # make sure that "self" is driving iteration
            # over image, i.e. when self is an fMRIImage and
            # outimage is an Image
            for dataslice in tmp:
                outimage.next(data=dataslice, value=tmp.itervalue)
        if hasattr(outimage, "close"): 
            outimage.close()
        return outimage


    def readall(self, clean=False): 
        """
        Read an entire Image object, returning a numpy, not another instance of
        Image. By default, it does not read 4d images. Missing values are
        filled in with the value of fill (default=self.fill=0.0).
        """
        value = self._source[self.grid.allslice()]
        if clean: 
            value = Image(N.nan_to_num(value, fill=self.fill))
        return value


    def check_grid(self, test): 
        return self.grid == test.grid



class ImageSequenceIterator(object):
    """
    Take a sequence of images, and an optional grid (which defaults to
    imgs[0].grid) and create an iterator whose next method returns array with
    shapes (len(imgs),) + self.imgs[0].next().shape Very useful for voxel-based
    methods, i.e. regression, one-sample t.
    """
    def __init__(self, imgs, grid=None):
        self.imgs = imgs
        if grid is None: self.grid = iter(self.imgs[0].grid)
        else: self.grid = iter(grid)

    def __iter__(self): 
        return self

    def next(self, value=None):
        if value is None: value = self.grid.next()
        v = [img.next(value=value) for img in self.imgs]
        return N.array(v, N.float64)

