"""
BaseImage class - wrapper for Image class to test changes to Image interface
"""
import sys, os, types

from attributes import attribute, readonly, deferto
import numpy as N

from neuroimaging import flatten
from neuroimaging.data import DataSource
from neuroimaging.image import Image
from neuroimaging.image.formats import getformat
from neuroimaging.reference import axis, mapping
from neuroimaging.reference.grid import SamplingGrid
from neuroimaging.reference.iterators import ParcelIterator, SliceParcelIterator


##############################################################################
class BaseImage(object):
    "Base class for all image objects.  Is constructed from an ndarray."

    #---------------------------------------------
    #   Attributes
    #---------------------------------------------
   
    class array (attribute):
        "raw data array"
        implements=N.ndarray

    class grid (attribute):
        "image sampling grid"
        implements=SamplingGrid
        def init(_,self): return SamplingGrid.identity(self.array.shape)

    class shape (readonly):
        "image shape (number of voxels in each dimension)"
        def get(_, self): return self.grid.shape

    class ndim (readonly):
        "number of image dimensions"
        def get(_, self): return len(self.shape)

    class postread (attribute):
        "apply this function to data after reading"
        default=lambda self, data: data

    deferto(array, ("__getitem__", "__setitem__", "put","compress"))
    deferto(grid, ("transform","itertype"))

    #---------------------------------------------
    #   Static Methods
    #---------------------------------------------

    @staticmethod
    def fromfile(filename, datasource=None):
        #return getreader(filename)(filename, datasource=datasource)
        from neuroimaging.refactoring.analyze import AnalyzeImage
        return AnalyzeImage(filename, datasource=datasource)

    #---------------------------------------------
    #   Instance Methods
    #---------------------------------------------

    #-------------------------------------------------------------------------
    def __init__(self, arr, grid=None):
        self.array = arr
        if grid is not None: self.grid = grid

    #-------------------------------------------------------------------------
    def __iter__(self):
        "Create an iterator over an image based on its grid's iterator."
        iter(self.grid)
        if isinstance(self.grid.iterator, ParcelIterator) or \
           isinstance(self.grid.iterator, SliceParcelIterator):
            flatten(self.array)
        return self

    #-------------------------------------------------------------------------
    def next(self, value=None, data=None):
        """
        The value argument here is used when, for instance one wants to
        iterate over one image with a ParcelIterator and write out data to
        this image without explicitly setting this image's grid to the
        original image's grid, i.e. to just take the value the original
        image's iterator returns and use it here.
        """
        if value is None: self.itervalue = value = self.grid.next()
        postread = getattr(self, 'postread', None) or (lambda x:x)

        if data is None:
            if self.itertype is 'slice':
                result = N.squeeze(self[value.slice])
            elif self.itertype is 'parcel':
                value.where.shape = N.product(value.where.shape)
                self.label = value.label
                result = self.compress(value.where, axis=0)
            elif self.itertype == 'slice/parcel':
                result = self[value.slice].compress(value.where)
            return self.postread(result)
        else:
            if self.itertype is 'slice':
                self.writeslice(value.slice, data)
            elif self.itertype in ("parcel","slice/parcel"):
                indices = N.nonzero(value.where.flatten())
                self.put(data, indices)

    #-------------------------------------------------------------------------
    def grid_array(self): 
        '''
        Read an entire Image object, returning a numpy array. By
        default, it does not read 4d images. 
        '''
        # NB - this used to be the readall method of the Image class
        # We may need port old code from this usage in due course
        return self[self.grid.iterator.allslice]
                
    #-------------------------------------------------------------------------
    def getvoxel(self, voxel):
        if len(voxel) != self.ndim:
            raise ValueError("expecting a voxel coordinate")
        return self[voxel]

    #-------------------------------------------------------------------------
    def write(self, filename, writer=None, clobber=False):
        "Write to file.  (was tofile)."
        #if writer is None: writer = getwriter(filename)
        from neuroimaging.refactoring.analyze import AnalyzeWriter
        writer = AnalyzeWriter().write
        writer(self, filename, clobber=clobber)


#-----------------------------------------------------------------------------
def image(input, datasource=DataSource(), grid=None):
    """
    Create a Image (volumetric image) object from either a file, an
    existing Image object, or an array.
    """
    
    # from array
    if isinstance(input, N.ndarray): return BaseImage(input, grid=grid)
        
    # from filename or url
    elif type(input) == types.StringType:
        return BaseImage.fromfile(input, datasource=datasource)


#-----------------------------------------------------------------------------
def writebrick(outfile, start, data, shape, offset=0, outtype=None,
  byteorder=sys.byteorder, return_tell = True):
    """from Utils """
    if return_tell:
        try: startpos = outfile.tell()
        except:
            outfile = file(outfile.name, 'rb+')
            startpos = 0
    if outtype: outdata = data.astype(outtype)
    else:
        outdata = data
        outtype = outdata.dtype
    if byteorder != sys.byteorder: outdata.byteswap()
    outdata.shape = (N.product(outdata.shape),)
    count = data.shape
    ndim = len(shape)

    # How many dimensions are "full" slices
    nslicedim = 0
    i = ndim - 1
    while count[i] is shape[i] and i >= 0:
        nslicedim = nslicedim + 1
        i = i - 1

    if nslicedim:
        nslice = N.product(shape[(ndim - nslicedim):])
    else:
        nslice = count[:-1]
        nslicedim = 1

    nloopdim = ndim - nslicedim
    test = N.product(N.less_equal(N.array(start) + N.array(count), N.array(shape)))
    if not test: raise ValueError, 'start+count not <= shape'
    nloop = N.product(count[nloopdim:])
    nskip = N.product(shape[nloopdim:])
    ntotal = N.product(count)
    elsize = outdata.dtype.itemsize
    shape_reverse = list(shape)
    shape_reverse.reverse()
    strides = [1] + list(N.multiply.accumulate(shape_reverse)[:-1])
    strides.reverse()
    strides = N.array(strides, N.int64)
    strides = strides * elsize
    outfile.seek(offset + N.add.reduce(start * strides))
    index = 0
    while index < ntotal:
        outdata[index:(index+nloop)].tofile(outfile)
        outfile.seek((nskip - nloop) * elsize, 1)
        index = index + nloop
    if return_tell:
        outfile.seek(startpos, 0)
    outfile.flush()
    del(outdata)


##############################################################################
class ImageSequenceIterator (object):
    """
    Take a sequence of images, and an optional grid (which defaults to
    imgs[0].grid) and create an iterator whose next method returns array
    with shapes (len(imgs),) + self.imgs[0].next().shape.  Very useful for
    voxel-based methods, i.e. regression, one-sample t.
    """
    def __init__(self, imgs, grid=None):
        self.imgs = imgs
        if grid is None: self.grid = iter(self.imgs[0].grid)
        else: self.grid = iter(grid)

    def __iter__(self): return self

    def next(self, value=None):
        if value is None: value = self.grid.next()
        v = [img.next(value=value) for img in self.imgs]
        return N.array(v, N.float64)
