import os, string, types
import options
import numpy as N
from slicer import Slicer
from reference import grid, pipes, axis
import enthought.traits as traits

if options.parallel:
    from Parallel import prange

if options.visual:
    import Plotting

spaceaxes = axis.space

class VImage(traits.HasTraits):

    nslicedim = traits.Int(2)
    shape = traits.ListInt()
    spatial_shape = traits.ListInt()
    start = traits.ListFloat()
    step = traits.ListFloat()

    def __init__(self, image, pipe=None, **keywords):
        '''
        Create a VImage (volumetric image) object from either a file, an existing VImage object, or an array.
        '''

        traits.HasTraits.__init__(self, **keywords)

        if isinstance(image, VImage):
            self.image = image.image
            self.isfile = image.isfile
        elif not pipe:
            if isinstance(image, ndarray):
                self.isfile = False
                self.image = pipes.ArrayPipe(image, **keywords)
            elif type(image) == types.StringType:
                self.isfile = True
                self.image = pipes.URLPipe(image, **keywords)
            elif type(image) in [types.ListType, types.TupleType]:
                self.image = pipes.ListPipe(image, **keywords)
                self.isfile = True # just in case any are files, we should try to open/close them
        else:
            self.image = pipe(image, **keywords)
            self.isfile = image.isfile
            
        self.type = type(self.image)

        # Find spatial grid -- this is the one that will be used generally

        self.grid = self.image.grid
        self.axes = self.grid.output_coords.axes
        self.shape = list(self.grid.shape)
        self.ndim = len(self.shape)

        self.axisnames = [dim.name for dim in self.axes]

        self.spatial_grid = Grid.linearize(self.grid).subset(spacedims)
        self.spatial_shape = list(self.spatial_grid.shape)

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
        
##     # Parallel iteration
    
##     try:
##         import mpi
##         parallel = options.image_parallel and options.parallel
##     except:
##         parallel = False
        
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
                if isend:
                    self.close()
                return return_value
            else:
                self.writeslice(slice, data)
                if isend:
                    self.close()

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
        '''Return a VImage instance that has an ArrayPipe as its image attribute.

        >>> from numpy import *
        >>> from BrainSTAT import *
        >>> test = VImage(testfile('anat+orig.HEAD'))
        >>> _test = test.toarray()
        >>> print _test.image.data.shape
        (124, 256, 256)
        >>> test = VImage(testfile('test_fmri.img'))
        >>> _test = test.toarray(slice=(2,), grid=test.grid)
        >>> print _test.shape
        (13, 128, 128)
        '''
        if self.isfile:
            self.close()
        if not keywords.has_key('grid'):
            keywords['grid'] = Grid.linearize(self.grid)

        if keywords.has_key('slice'):
            whichdim = keywords['grid'].output_coords.axes[len(keywords['slice']):]
            whichdim = [dim.name for dim in whichdim]
            keywords['grid'] = keywords['grid'].subset(whichdim)

        if clean:
            _clean = nan_to_num
            _data = _clean(self.readall(**keywords))
        else:
            _data = self.readall(**keywords)
        return VImage(_data, **keywords)

    def tofile(self, filename, array=True, **keywords):
        outimage = VImage(filename, template=self, mode='w', **keywords)

        if array:
            tmp = self.toarray(**keywords)
            outimage.image.write((0,)*tmp.ndim, tmp.image.data)
        else:
            tmp = iter(self)
            outimage = iter(outimage)
            for dataslice in tmp:
                outimage.next(data=dataslice)
                del(dataslice); gc.collect()
                
        outimage.close()
        
        return outimage

    fill = traits.Float(0.0)
    
    def readall(self, only3d=False, try4d=True, clean=False, **keywords): 
        '''
        Read an entire VImage object, returning a numpy, not another instance of VImage. By default, it does not read 4d images. Missing values are filled in with the value of fill (default=self.fill=0.0).
        '''

        if self.ndim > 3 and only3d and not try4d and not keywords.has_key('slice'):
            raise ValueError, 'readall expecting 3d image -- set only3d to False for non 3d image, or set slice'
        elif keywords.has_key('slice'):
            _slice = keywords.pop('slice')
            value = self.getslice(_slice, **keywords)
        else:
            value = self.image.read((0,) * self.ndim, self.image.shape)                
        if clean:
            value = VImage(_clean(value, fill=self.fill))
        return value

    def getslice(self, slice, **keywords):

        start = slice[0:len(slice)]
        count = (1,) * len(slice)
        return_shape = ()
        for i in range(self.ndim - len(slice)):
            start = start + (0,)
            count = count + (self.shape[i+len(slice)],)
            return_shape = return_shape + (self.shape[i+len(slice)],)
        tmp = self.image.read(start, count, **keywords).astype(N.Float)
        tmp.shape = return_shape
        return tmp

    def writeslice(self, slice, data, mode='r+', **keywords):
        start = slice[0:len(slice)]
        for i in range(self.ndim - len(slice)):
            start = start + (0,)
        outshape = tuple(N.ones((len(slice),))) + data.shape
        if data.shape != outshape:
            reshapedata = reshape(data, (1,) * len(slice) + data.shape)
        else:
            reshapedata = data
        self.image.write(start, reshapedata, **keywords)
        del(reshapedata)

    def check_grid(self, test):
        return self.grid == test.grid

class fMRIImage(VImage):
    TR = traits.Float(2.0)

    def __iter__(self, mode='r'):
        self.nloopdim = self.ndim - 1 - self.nslicedim

        if VImage.parallel:
            a, b = prange(self.shape[1])
        else:
            a = 0
            b = self.shape[1]

        self.slicer = iter(Slicer((b-a,) + tuple(self.shape[2:]), nloopdim = self.nloopdim, shift=a))
        return self

    def next(self, data = None):
        self.slice, isend = self.slicer.next()
        return_value = N.zeros([self.shape[0]] + self.shape[-self.nslicedim:], N.Float)
        if data is None:
            for i in range(self.shape[0]):
                return_value[i] = self.getslice((i,) + self.slice)
            if isend:
                self.close()
            return return_value
        else:
            for i in range(self.shape[0]):
                self.writeslice((i,) + self.slice, data[i])
            if isend:
                self.close()
            return None

    def tofile(self, filename, **keywords):
        VImage.tofile(self, filename, array=False, **keywords)
        
    def frame(self, i, **keywords):
        return self.toarray(slice=(i,))

    def timeseries(self, voxel, **keywords):
        timeseries = N.zeros((self.shape[0],), N.Float)
        for i in range(self.shape[0]):
            timeseries[i] = float(self.getslice((i,) + tuple(voxel), **keywords))
        return timeseries

    if options.visual:
        import pylab
        def plotslice(self, slice, time=0, **keywords):
            frame = self.frame(time, **keywords)
            return frame.plotslice(slice)

        def plotseries(self, voxel, **extra):
            plotdata = self.timeseries(voxel, **extra)
            pylab.plot(self.time.values(), plotdata, **extra)

