import numpy as N

from neuroimaging import reverse
from neuroimaging.core.reference.mapping import Mapping, Affine, DegenerateAffine
from neuroimaging.core.reference.axis import space, RegularAxis, VoxelAxis, Axis
from neuroimaging.core.reference.coordinate_system import VoxelCoordinateSystem,\
  DiagonalCoordinateSystem, CoordinateSystem
from neuroimaging.core.reference.iterators import SliceIterator,\
  ParcelIterator, SliceParcelIterator


class SamplingGrid (object):

    @staticmethod
    def from_start_step(names=space, shape=(), start=(), step=()): 
        """
        Create a SamplingGrid instance from sequences of names, shape, start
        and step.
        """
        ndim = len(names)
        # fill in default step size
        step = N.asarray(step)
        step = N.where(step, step, 1.)

        indim = [VoxelAxis(name=names[i], length=shape[i]) for i in range(ndim)]
        input_coords = VoxelCoordinateSystem('voxel', indim)

        outdim = [RegularAxis(name=names[i], length=shape[i],
          start=start[i], step=step[i]) for i in range(ndim)]
        output_coords = DiagonalCoordinateSystem('world', outdim)

        transform = output_coords.transform()
        mapping = Affine(input_coords, output_coords, transform)
        return SamplingGrid(shape=list(shape), mapping=mapping)


    @staticmethod
    def identity(shape=(), names=space):
        """
        return an identity grid of the given shape.
        """
        ndim = len(shape)
        if len(names) != ndim:
            raise ValueError('shape and number of axisn ames do not agree')
        w = Affine.identity(ndim, names=names)
        return SamplingGrid(shape=list(shape), mapping=w)


    def __init__(self, shape, mapping):
        # These guys define the structure of the grid.
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.mapping = mapping
        self.input_coords = mapping.input_coords
        self.output_coords = mapping.output_coords

        # These guys are for use of the SamplingGrid as an iterator.
        self._axis = 0
        self._itertype = "slice"
        self._parcelseq = None
        self._parcelmap = None

    def allslice (self):
        """
        a slice object representing the entire grid
        """
        try: 
            return self.iterator.allslice
        except AttributeError: 
            return slice(0, self.shape[0])

    def range(self):
        """
        return the coordinate values in the same format as numpy.indices.
        """
        indices = N.indices(self.shape)
        tmp_shape = indices.shape
        # reshape indices to be a sequence of coordinates
        indices.shape = (self.mapping.ndim(), N.product(self.shape))
        _range = self.mapping(indices)
        _range.shape = tmp_shape
        return _range 

    def __iter__(self):
        itermethod = {
          "slice": self._iterslices,
          "parcel": self._iterparcels,
          "slice/parcel": self._itersliceparcels
        }.get(self._itertype)
        if itermethod is None:
            raise ValueError("unknown itertype %s"%`self._itertype`)
        return itermethod()

    def set_iter(self, itertype="slice", parcelmap=None, parcelseq=None, axis=None):
        self._itertype = itertype
        if parcelmap is not None:
            self._parcelmap = parcelmap
        if parcelseq is not None:
            self._parcelseq = parcelseq
        if axis is not None:
            self._axis = axis

    def _iterslices(self):
        self.iterator = SliceIterator(self.shape, axis=self._axis)
        return self

    def _iterparcels(self):
        self.iterator = ParcelIterator(self._parcelmap, self._parcelseq)
        return self


    def _itersliceparcels(self):
        self.iterator = SliceParcelIterator(self._parcelmap, self._parcelseq)
        return self


    def next(self):
        self.itervalue = self.iterator.next()
        return self.itervalue


    def slab(self, start, step, count, axis=0):
        """
        A sampling grid for a hyperslab of data from an array, i.e.
        what would be output from a subsampling of every 2nd voxel or so.

        By default, the iterator of the slab is a SliceIterator
        with the same start, step, count and iterating over the
        specified axis with nslicedim=1.
        """

        if isinstance(self.mapping, Affine):
            ndim = self.ndim
            T = 0 * self.mapping.transform
            T[0:ndim] = self.mapping(start)
            T[ndim, ndim] = 1.
            for i in range(ndim):
                v = N.zeros((ndim,))
                w = 1 * v
                v[i] = step[i]
                T[0:ndim,i] = self.mapping(v) - self.mapping(w)
            _map = Affine(self.mapping.input_coords,
                          self.mapping.output_coords, T)
        else:
            def __map(x, start=start, step=step, _f=self.mapping):
                v = start + step * x
                return _f(v)
            _map = Mapping(self.mapping.input_coords,
                           self.mapping.output_coords, __map)

        g = SamplingGrid(shape=count, mapping=_map)
        g.end = N.array(start) + N.array(count) * N.array(step)
        g.start = start
        g.step = step
        g.axis = axis
        g._itertype = 'slice'
        return iter(g)


    def transform(self, matrix): 
        self.mapping = matrix * self.mapping

    def matlab2python(self):
        return SamplingGrid(shape=reverse(self.shape),
          mapping=self.mapping.matlab2python())

    def python2matlab(self):
        return SamplingGrid(shape=reverse(self.shape),
          mapping=self.mapping.python2matlab())

    def replicate(self, n, concataxis="concat"):
        """
        Duplicate self n times, returning a ConcatenatedGrids with
        shape == (n,)+self.shape.
        """
        return ConcatenatedIdenticalGrids(self, n, concataxis=concataxis)

class ConcatenatedGrids(SamplingGrid):
    """
    Return a grid formed by concatenating a sequence of grids. Checks are done
    to ensure that the coordinate systems are consistent, as is the shape.
    It returns a grid with the proper shape but no inverse.
    This is most likely the kind of grid to be used for fMRI images.
    """

    def _grids (self, grids):
        # check mappings are affine
        check = N.sum([not isinstance(grid.mapping, Affine)\
                          for grid in grids])
        if check: raise ValueError('must all be affine mappings!')

        # check shapes are identical
        s = grids[0].shape
        check = N.sum([grid.shape != s for grid in grids])
        if check: raise ValueError('subgrids must have same shape')

        # check input coordinate systems are identical
        ic = grids[0].mapping.input_coords
        check = N.sum([grid.mapping.input_coords != ic\
                           for grid in grids])
        if check: raise ValueError(
              'subgrids must have same input coordinate systems')

        # check output coordinate systems are identical
        oc = grids[0].mapping.output_coords
        check = N.sum([grid.mapping.output_coords != oc\
                           for grid in grids])
        if check: raise ValueError(
              'subgrids must have same output coordinate systems')
        return tuple(grids)

    def _mapping(self):
        def mapfunc(x):
            try:
                I = x[0].view(N.int32)
                X = x[1:]
                v = N.zeros(x.shape[1:], N.float64)
                for j in I.shape[0]:
                    v[j] = self.grids[I[j]].mapping(X[j])
                return v
            except:
                i = int(x[0])
                x = x[1:]
                return self.grids[i].mapping(x)
                
        newaxis = Axis(name=self.concataxis)
        ic = self.grids[0].mapping.input_coords
        newin = CoordinateSystem(
              '%s:%s'%(ic.name, self.concataxis), [newaxis] + list(ic.axes()))
        oc = self.grids[0].mapping.output_coords
        newout = CoordinateSystem(
              '%s:%s'%(oc.name, self.concataxis), [newaxis] + list(oc.axes()))
        return Mapping(newin, newout, mapfunc)


    def __init__(self, grids, concataxis="concat"):
        self.grids = self._grids(grids)
        self.concataxis = concataxis
        mapping = self._mapping()
        shape = (len(self.grids),) + self.grids[0].shape
        SamplingGrid.__init__(self, shape, mapping)

    def subgrid(self, i): 
        return self.grids[i]

class ConcatenatedIdenticalGrids(ConcatenatedGrids):

    def __init__(self, grid, n, concataxis="concat"):
        ConcatenatedGrids.__init__(self, [grid for i in range(n)], concataxis)
        self.mapping = self._mapping()

    def _mapping(self):
        newaxis = Axis(name=self.concataxis)
        ic = self.grids[0].mapping.input_coords
        newin = CoordinateSystem(
            '%s:%s'%(ic.name, self.concataxis), [newaxis] + list(ic.axes()))
        oc = self.grids[0].mapping.output_coords
        newout = CoordinateSystem(
            '%s:%s'%(oc.name, self.concataxis), [newaxis] + list(oc.axes()))

        t = self.grids[0].mapping.transform
        ndim = t.shape[0]-1
        T = N.zeros((ndim+2,)*2, N.float64)
        T[0:ndim,0:ndim] = t[0:ndim,0:ndim]
        T[0:ndim,-1] = t[0:ndim,-1]
        T[ndim,ndim] = 1.
        T[(ndim+1),(ndim+1)] = 1.
        return Affine(newin, newout, T)

