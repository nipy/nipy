import neuroimaging.image as image
import neuroimaging.reference.grid as grid
from neuroimaging.reference.grid_iterators import SliceIteratorNext

import neuroimaging.reference.warp as warp
import neuroimaging.reference.coordinate_system as coordinate_system
import enthought.traits as traits
import numpy as N

class fMRIListWarp(warp.Warp):

    def __init__(self, input_coords, output_coords, maps, **keywords):
        self._maps = maps

    def map(self, coords, inverse=False):
        if len(coords.shape) > 1:
            n = coords.shape[1]
            value = []
            for i in range(n):
                value.append(self._maps[coords[i][0]](coords[i][1:]))
        else:
            return self._maps[coords[0]][coords[1:]]

class fMRISliceIterator(grid.SliceIterator):
    """
    Instead of iterating over slices of a 4d file -- return slices
    of timeseries.
    """

    nframe = traits.Int()

    def __init__(self, shape, **keywords):
        grid.SliceIterator.__init__(self, shape[1:], **keywords)
        self.nframe = shape[0]

    def next(self):
        value = grid.SliceIterator.next(self)
        _slice = [slice(0,self.nframe,1), value.slice]
        return SliceIteratorNext(slice=_slice, type='slice')

class fMRIParcelIterator(grid.ParcelIterator):
    """
    Return parcels of timeseries.
    """

    nframe = traits.Int()

    def __init__(self, shape, labels, **keywords):
        grid.ParcelIterator.__init__(self, shape, labels, **keywords)
        self.nframe = shape[0]

class fMRISamplingGrid(grid.SamplingGrid):

    def __iter__(self):
        if self.itertype is 'slice':
            self.iterator = iter(fMRISliceIterator(shape=self.shape))
        if self.itertype is 'parcel':
            self.iterator = iter(fMRIParcelIterator(self.shape, self.labels))
        return self

    def subgrid(self, i):
        """
        Return a subgrid of fMRISamplingGrid. If the image's warp is an
        Affine instance and is \'diagonal\' in time, then it returns
        a new Affine instance. Otherwise, if the image's warp is a list of
        warps, it returns the i-th warp.
        Finally, if these two do not hold, it returns a generic, non-invertible
        map in the original output coordinate system.
        """

        tol = 1.0e-07

        if isinstance(self.warp, warp.Affine):
            n = len(self.shape)
            t = self.warp.transform
            offdiag = N.add.reduce(t[1:n,0]**2) + N.add.reduce(t[0,1:n]**2)
            norm = N.add.reduce(N.add.reduce(t**2))
            if N.sqrt(offdiag / norm) < tol:
                isaffine = True
            else:
                isaffine = False
        else:
            isaffine = False

        inaxes = self.warp.input_coords.axes[1:]
        incoords = coordinate_system.CoordinateSystem(self.warp.input_coords.name+'-subgrid', inaxes)

        if isinstance(self.warp, fMRIListWarp):
            outaxes = self.warp.output_coords.axes[1:]
            outcoords = coordinate_system.CoordinateSystem(self.warp.output_coords.name, outaxes)        

            W = warp.Affine(incoords, outcoords, self._maps[i])

        elif isaffine:

            outaxes = self.warp.output_coords.axes[1:]
            outcoords = coordinate_system.CoordinateSystem(self.warp.output_coords.name, outaxes)        

            t = t[1:,1:]
            W = warp.Affine(incoords, outcoords, t)
        else:

            outaxes = self.warp.output_coords.axes[1:]
            outcoords = coordinate_system.CoordinateSystem(self.warp.output_coords.name, outaxes)        

            def _map(x, fn=self.warp.map, **keywords):
                if len(x.shape) > 1:
                    _x = N.zeros((x.shape[0]+1,) + x.shape[1:], N.Float)
                else:
                    _x = N.zeros((x.shape[0]+1,), N.Float)
                _x[0] = i
                return fn(_x)

            W = warp.Warp(incoords, outcoords, _map)
        return grid.SamplingGrid(shape=self.shape[1:], warp=W)

class fMRIImage(image.Image):
    frametimes = traits.Any()
    slicetimes = traits.Any()

    def __init__(self, _image, **keywords):
        image.Image.__init__(self, _image, **keywords)
        self.grid = fMRISamplingGrid(warp=self.grid.warp, shape=self.grid.shape)

    def tofile(self, filename, **keywords):
        image.Image.tofile(self, filename, array=False, **keywords)
        
    def frame(self, i, **keywords):
        return self.toarray(slice=(slice(i)))

    def next(self, data=None, callgrid=True, type=None):

        value = self.grid.next()

        if type is None:
            type = value.type

        if type is 'slice':
            if data is None:
                return_value = N.squeeze(self.getslice(value.slice))
                return return_value
            else:
                self.writeslice(value.slice, data)

        elif type is 'parcel':
            if data is None:
                value.where.shape = N.product(value.where.shape)
                self.label = value.label
                return self.buffer.compress(value.where, axis=1)
            else:
                indices = N.nonzero(value.where)
                self.buffer.put(data, indices)
                self.buffer.sync()

    def __iter__(self):
        """
        Create an iterator over an image based on its grid's iterator.
        """
        iter(self.grid)

        if self.grid.itertype is 'parcel':
            self.buffer = self.readall()
            self.buffer.shape = (self.buffer.shape[0], N.product(self.buffer.shape[1:]))
        return self


