from neuroimaging import traits
import numpy as N

from neuroimaging import flatten
from neuroimaging.core.image.image import Image
from neuroimaging.core.reference.coordinate_system import CoordinateSystem
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.reference.iterators import ParcelIterator
from neuroimaging.core.reference.mapping import Mapping, Affine

from neuroimaging.core.reference.iterators import SliceIterator

class fMRISamplingGrid(SamplingGrid):

    def __init__(self, shape, mapping, input_coords, output_coords):
        SamplingGrid.__init__(self, shape, mapping, input_coords, output_coords)


    def isproduct(self, tol = 1.0e-07):
        "Determine whether the affine   ation is 'diagonal' in time."

        if not isinstance(self.mapping, Affine):
            return False
        ndim = self.ndim
        t = self.mapping.transform
        offdiag = N.add.reduce(t[1:ndim,0]**2) + N.add.reduce(t[0,1:ndim]**2)
        norm = N.add.reduce(N.add.reduce(t**2))
        return N.sqrt(offdiag / norm) < tol


    def subgrid(self, i):
        """
        Return a subgrid of fMRISamplingGrid. If the image's mapping is an
        Affine instance and is 'diagonal' in time, then it returns a new
        Affine instance. Otherwise, if the image's mapping is a list of
        mappings, it returns the i-th mapping.  Finally, if these two do not
        hold, it returns a generic, non-invertible map in the original output
        coordinate system.
        """
        # TODO: this bit should be handled by CoordinateSystem,
        # eg: incoords = self.mapping.input_coords.subcoords(...)
        incoords = CoordinateSystem(
          self.input_coords.name+'-subgrid',
          self.input_coords.axes()[1:])

        outaxes = self.output_coords.axes()[1:]
        outcoords = CoordinateSystem(
            self.output_coords.name, outaxes)        


        if self.isproduct():
            t = self.mapping.transform
            t = t[1:,1:]
            W = Affine(t)

        else:
            def _map(x, fn=self.mapping.map, **keywords):
                if len(x.shape) > 1:
                    _x = N.zeros((x.shape[0]+1,) + x.shape[1:], N.float64)
                else:
                    _x = N.zeros((x.shape[0]+1,), N.float64)
                _x[0] = i
                return fn(_x)
            W = Mapping(_map)

        _grid = SamplingGrid(self.shape[1:], W, incoords, outcoords)
        return _grid



class fMRIImage(Image):

    def __init__(self, _image, **keywords):
        Image.__init__(self, _image, **keywords)
        self.frametimes = keywords.get('frametimes', None)
        self.slicetimes = keywords.get('slicetimes', None)

        self.grid = fMRISamplingGrid(self.grid.shape, self.grid.mapping, self.grid.input_coords, self.grid.output_coords)
        if self.grid.isproduct():
            ndim = len(self.grid.shape)
            n = [self.grid.input_coords.axisnames()[i] \
                 for i in range(ndim)]
            d = n.index('time')
            transform = self.grid.mapping.transform[d, d]
            start = self.grid.mapping.transform[d, ndim]
            self.frametimes = start + N.arange(self.grid.shape[d]) * transform


    def frame(self, i, clean=False, **keywords):
        data = N.squeeze(self[slice(i,i+1)])
        if clean: data = N.nan_to_num(data)
        return Image(data, grid=self.grid.subgrid(i), **keywords)


    def slice_iterator(self, mode='r', axis=1):
        ''' Return slice iterator for this image. By default we iterate
        over the C{axis=1} instead of C{axis=0} as for the L{Image} class.

        @param axis: The index of the axis (or axes) to be iterated over. If
            a list is supplied, the axes are iterated over slowest to fastest.
        @type axis: C{int} or C{list} of C{int}.
        @param mode: The mode to run the iterator in.
            'r' - read-only (default)
            'w' - read-write
        @type mode: C{string}
        '''
        return SliceIterator(self, mode=mode, axis=axis)

    def from_slice_iterator(self, other, axis=1):
        """
        Take an existing L{SliceIterator} and use it to set the values
        in this image. By default we iterate over the C{axis=1} for this image
        instead of C{axis=0} as for the L{Image} class.

        @param other: The iterator from which to take the values
        @type other: L{SliceIterator}
        @param axis: The axis to iterate over for this image.
        @type axis: C{int} or C{list} of {int}
        """
        it = iter(SliceIterator(self, mode='w', axis=axis))
        for s in other:
            it.next().set(s)
