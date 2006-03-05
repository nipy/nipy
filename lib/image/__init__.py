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
            if isinstance(image, N.ndarray):
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

##         self.spatial_grid = Grid.linearize(self.grid).subset(spacedims)
##         self.spatial_shape = list(self.spatial_grid.shape)

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
        return self

    def next(self, data=None, callgrid=True, type=None):
        if callgrid:
            self._itervalue = self.grid.next()

        value = self._itervalue

        if type is None:
            type = value.type

        if type is 'slice':
            slice = value.slice
            if data is None:
                return_value = self.getslice(slice)
                return return_value
            else:
                self.writeslice(slice, data)

        elif type is 'labelled slice':
            if value.newslice:
                if data is None:
                    self.buffer = self.next(self, type='slice')
                    self.buffer.shape = self._bufshape
                else:
                    self.buffer = self.fill * self.buffer
            if value.keep:
                if data is None:
                    return compress(value.keep, self.buffer)
                else:
                    self.buffer[value.keep] = data
            else:
                if data is not None:
                    self.buffer.shape = self._outshape
                    self.next(self, type='slice', data=self.buffer, callgrid=False)
                    self.buffer.shape = self._bufshape

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
    
    def readall(self, try4d=True, clean=False, **keywords): 
        '''
        Read an entire Image object, returning a numpy, not another instance of Image. By default, it does not read 4d images. Missing values are filled in with the value of fill (default=self.fill=0.0).
        '''

        if self.ndim > 3 and not try4d:
            if not keywords.has_key('slice'):
                raise ValueError, 'readall expecting 3d image -- set try4d to True for non 3d image, or set slice=slice'
        if keywords.has_key('slice'):
            _slice = keywords['slice']
        else:
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

## class fMRIImage(Image):
##     TR = traits.Float(2.0)

##     def __iter__(self, mode='r'):
##         self.nloopdim = self.ndim - 1 - self.nslicedim

## ##        self.slicer = iter(Slicer((b-a,) + tuple(self.shape[2:]), nloopdim = self.nloopdim, shift=a))
##         return self

##     def next(self, data = None):
##         self.slice, isend = self.slicer.next()
##         return_value = N.zeros([self.shape[0]] + self.shape[-self.nslicedim:], N.Float)
##         if data is None:
##             for i in range(self.shape[0]):
##                 return_value[i] = self.getslice((i,) + self.slice)
##             if isend:
##                 self.close()
##             return return_value
##         else:
##             for i in range(self.shape[0]):
##                 self.writeslice((i,) + self.slice, data[i])
##             if isend:
##                 self.close()
##             return None

##     def tofile(self, filename, **keywords):
##         Image.tofile(self, filename, array=False, **keywords)
        
##     def frame(self, i, **keywords):
##         return self.toarray(slice=(i,))

##     def timeseries(self, voxel, **keywords):
##         timeseries = N.zeros((self.shape[0],), N.Float)
##         for i in range(self.shape[0]):
##             timeseries[i] = float(self.getslice((i,) + tuple(voxel), **keywords))
##         return timeseries

