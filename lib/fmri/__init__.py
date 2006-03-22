import neuroimaging.image as image
import neuroimaging.reference.grid as grid

import neuroimaging.reference.mapping as mapping
import neuroimaging.reference.coordinate_system as coordinate_system
import enthought.traits as traits
import numpy as N
from iterators import fMRISliceIterator, fMRIParcelIterator, fMRISliceParcelIterator

import hrf, filters

class fMRIListMapping(mapping.Mapping):

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

class fMRISamplingGrid(grid.SamplingGrid):

    def __init__(self, **keywords):
        grid.SamplingGrid.__init__(self, **keywords)

    def __iter__(self):

        if self.itertype == 'slice':
            self.iterator = iter(fMRISliceIterator(shape=self.shape))
        elif self.itertype == 'parcel':
            self.iterator = iter(fMRIParcelIterator(self.labels, self.labelset))
        elif self.itertype == 'slice/parcel':
            self.iterator = iter(fMRISliceParcelIterator(self.labels, self.labelset, self.shape))
        return self

    def isproduct(self, tol = 1.0e-07):
        """
        Determine whether the affine transformation is \'diagonal\' in time.
        """

        if isinstance(self.mapping, mapping.Affine):
            n = len(self.shape)
            t = self.mapping.transform
            offdiag = N.add.reduce(t[1:n,0]**2) + N.add.reduce(t[0,1:n]**2)
            norm = N.add.reduce(N.add.reduce(t**2))
            if N.sqrt(offdiag / norm) < tol:
                return True
            else:
                return False
        else:
            return False

    def subgrid(self, i):
        """
        Return a subgrid of fMRISamplingGrid. If the image's mapping is an
        Affine instance and is \'diagonal\' in time, then it returns
        a new Affine instance. Otherwise, if the image's mapping is a list of
        mappings, it returns the i-th mapping.
        Finally, if these two do not hold, it returns a generic, non-invertible
        map in the original output coordinate system.
        """


        inaxes = self.mapping.input_coords.axes[1:]
        incoords = coordinate_system.CoordinateSystem(self.mapping.input_coords.name+'-subgrid', inaxes)

        if isinstance(self.mapping, fMRIListMapping):
            outaxes = self.mapping.output_coords.axes[1:]
            outcoords = coordinate_system.CoordinateSystem(self.mapping.output_coords.name, outaxes)        

            W = mapping.Affine(incoords, outcoords, self._maps[i])
        elif self.isproduct():
            outaxes = self.mapping.output_coords.axes[1:]
            outcoords = coordinate_system.CoordinateSystem(self.mapping.output_coords.name, outaxes)        

            t = self.mapping.transform
            t = t[1:,1:]
            W = mapping.Affine(incoords, outcoords, t)
        else:
            outaxes = self.mapping.output_coords.axes[1:]
            outcoords = coordinate_system.CoordinateSystem(self.mapping.output_coords.name, outaxes)        

            def _map(x, fn=self.mapping.map, **keywords):
                if len(x.shape) > 1:
                    _x = N.zeros((x.shape[0]+1,) + x.shape[1:], N.Float)
                else:
                    _x = N.zeros((x.shape[0]+1,), N.Float)
                _x[0] = i
                return fn(_x)

            W = mapping.Mapping(incoords, outcoords, _map)

        _grid = grid.SamplingGrid(shape=self.shape[1:], mapping=W)
        _grid.itertype = self.itertype
        _grid.labels = self.labels
        _grid.labelset = self.labelset
        return _grid

class fMRIImage(image.Image):
    frametimes = traits.Any()
    slicetimes = traits.Any()
    TR = traits.Any()

    def __init__(self, _image, **keywords):
        image.Image.__init__(self, _image, **keywords)
        self.grid = fMRISamplingGrid(mapping=self.grid.mapping, shape=self.grid.shape)
        if self.grid.isproduct():
            ndim = len(self.grid.shape)
            n = [self.grid.mapping.input_coords.axisnames[i] for i in range(ndim)]
            d = n.index('time')
            self.TR = self.grid.mapping.transform[d, d] # 
            start = self.grid.mapping.transform[d, ndim]
            self.frametimes = start + N.arange(self.grid.shape[d]) * self.TR

    def tofile(self, filename, **keywords):
        image.Image.tofile(self, filename, array=False, **keywords)
        
    def frame(self, i, **keywords):
        return self.toarray(slice=(slice(i)))

    def next(self, value=None, data=None):
        """
        The value argument here is used when, for instance one wants to
        iterate over one image with a ParcelIterator and write out data
        to this image without explicitly setting this image's grid to
        the original image's grid, i.e. to just take the value the
        original image's iterator returns and use it here.
        """
        if value is None:
            self.itervalue = self.grid.next()
            value = self.itervalue

        itertype = value.type

        if itertype == 'slice':
            if data is None:
                return_value = N.squeeze(self.getslice(value.slice))
                return return_value
            else:
                self.writeslice(value.slice, data)

        elif itertype == 'parcel':
            if data is None:
                value.where.shape = N.product(value.where.shape)
                self.label = value.label
                return self.buffer.compress(value.where, axis=1)
            else:
                for i in range(self.grid.shape[0]):
                    _buffer = self.getslice([slice(i,i+1)])
                    _buffer.put(data, indices)

        elif itertype == 'slice/parcel':
            if data is None:
                value.where.shape = N.product(value.where.shape)
                self.label = value.label
                tmp = self.getslice(value.slice)
                tmp.shape = (tmp.shape[0], N.product(tmp.shape[1:]))
                return tmp.compress(value.where, axis=1)
            else:
                indices = N.nonzero(value.where)
                _buffer = self.getslice(value.slice)
                _buffer.put(data, indices)

    def __iter__(self):
        """
        Create an iterator over an image based on its grid's iterator.
        """
        iter(self.grid)

        if self.grid.itertype is 'parcel':
            self.buffer = self.readall()
            self.buffer.shape = (self.buffer.shape[0], N.product(self.buffer.shape[1:]))
        return self


