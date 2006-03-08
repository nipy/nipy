import os, string, types
import numpy as N
from neuroimaging.reference import grid, axis
import utils
import pipes
import enthought.traits as traits

spaceaxes = axis.space

class Image(traits.HasTraits):

    nslicedim = traits.Int(2)
    shape = traits.ListInt()
    spatial_shape = traits.ListInt()
    start = traits.ListFloat()
    step = traits.ListFloat()

    def __init__(self, image, pipe=None, **keywords):
        '''
        Create a Image (volumetric image) object from either a file, an existing Image object, or an array.
        '''

        traits.HasTraits.__init__(self, **keywords)

        if isinstance(image, Image):
            self.image = image.image
            self.isfile = image.isfile
        elif not pipe:
            if isinstance(image, N.ndarray) or isinstance(image, N.core.memmap):
                self.isfile = False
                self.image = pipes.ArrayPipe(image, **keywords)
            elif type(image) == types.StringType:
                self.isfile = True
                pipe = pipes.URLPipe(image, **keywords)
                self.image = pipe.getimage()
##             elif type(image) in [types.ListType, types.TupleType]:
##                 self.image = pipes.ListPipe(image, **keywords)
##                 self.isfile = True # just in case any are files, we should try to open/close them
        else:
            self.image = pipe(image, **keywords)
            self.isfile = image.isfile
            
        self.type = type(self.image)

        # Find spatial grid -- this is the one that will be used generally

        self.grid = self.image.grid
        self.shape = list(self.grid.shape)
        self.ndim = len(self.shape)

    def __del__(self):
        if self.isfile:
            try:
                self.image.close()
            except:
                pass
        else:
            del(self.image)

    def open(self, mode='r'):
        if self.isfile:
            self.image.open(mode=mode)

    def close(self):
        if self.isfile:
            try:
                self.image.close()
            except:
                pass
        
    def __iter__(self):
        """
        Create an iterator over an image based on its grid's iterator.
        """
        iter(self.grid)

        if isinstance(self.grid.iterator, grid.ParcelIterator):
            if hasattr(self.image, 'memmap'):
                self.buffer = self.image.memmap
            else:
                self.buffer = self.readall()
            self.buffer.shape = N.product(self.buffer.shape)
        return self

    def next(self, value=None, data=None):
        """
        The value argument here is used when, for instance one wants to
        iterate over one image with a ParcelIterator and write out data
        to this image without explicitly setting this image\'s grid to
        the original image\'s grid, i.e. to just take the value the
        original image\'s iterator returns and use it here.
        """
        if value is None:
            self.itervalue = self.grid.next()
            value = self.itervalue

        itertype = value.type

        if itertype is 'slice':
            if data is None:
                return_value = N.squeeze(self.getslice(value.slice))
                return return_value
            else:
                self.writeslice(value.slice, data)

        elif itertype is 'parcel':
            if data is None:
                value.where.shape = N.product(value.where.shape)
                self.label = value.label
                return self.buffer.compress(value.where, axis=0)
            else:
                indices = N.nonzero(value.where)
                self.buffer.put(data, indices)
                try:
                    self.buffer.sync()
                except:
                    pass

    def getvoxel(self, voxel):
        if len(voxel) != self.ndim:
            raise ValueError, 'expecting a voxel coordinate'
        return self.getslice(voxel)

    def toarray(self, clean=True, **keywords):
        '''Return a Image instance that has an ArrayPipe as its image attribute.

        >>> from numpy import *
        >>> from BrainSTAT import *
        >>> test = Image(testfile('anat+orig.HEAD'))
        >>> _test = test.toarray()
        >>> print _test.image.data.shape
        (124, 256, 256)
        >>> test = Image(testfile('test_fmri.img'))
        >>> _test = test.toarray(slice=(2,), grid=test.grid)
        >>> print _test.shape
        (13, 128, 128)
        '''

        if self.isfile:
            self.close()

        if clean:
            _clean = N.nan_to_num
            _data = _clean(self.readall(**keywords))
        else:
            _data = self.readall(**keywords)
        return Image(_data, grid=self.grid, **keywords)

    def tofile(self, filename, array=True, **keywords):
        outimage = Image(filename, mode='w', grid=self.grid, **keywords)

        if array:
            tmp = self.toarray(**keywords)
            outimage.image.writeslice(slice(0,self.grid.shape[0],1), tmp.image.data)
        else:
            tmp = iter(self)
            outimage = iter(outimage)
            for dataslice in tmp:
                outimage.next(data=dataslice)
                del(dataslice); gc.collect()
                
        outimage.close()
        
        return outimage

    fill = traits.Float(0.0)
    
    def readall(self, clean=False, **keywords): 
        '''
        Read an entire Image object, returning a numpy, not another instance of Image. By default, it does not read 4d images. Missing values are filled in with the value of fill (default=self.fill=0.0).
        '''

        _slice = slice(0, self.grid.shape[0], 1)
        value = self.image.getslice(_slice)

        if clean:
            value = Image(_clean(value, fill=self.fill))
        return value

    def getslice(self, slice):
        return self.image.getslice(slice)

    def writeslice(self, slice, data):
        return self.image.writeslice(slice, data)

    def check_grid(self, test):
        return self.grid == test.grid

