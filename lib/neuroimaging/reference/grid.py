import numpy as N

from attributes import attribute, readonly, enum, deferto, clone
from protocols import Iterator

from neuroimaging import reverse
from neuroimaging.reference.mapping import Mapping, Affine, DegenerateAffine
from neuroimaging.reference.axis import space, RegularAxis, VoxelAxis, Axis
from neuroimaging.reference.coordinate_system import VoxelCoordinateSystem,\
  DiagonalCoordinateSystem, CoordinateSystem
from neuroimaging.reference.iterators import itertypes, SliceIterator,\
  ParcelIterator, SliceParcelIterator



class SamplingGrid (object):

    class shape (readonly): implements=tuple
    class ndim (readonly): get=lambda _, self: len(self.shape)
    class mapping (attribute): implements=Mapping
    class iterator (attribute): implements=Iterator
    class itertype (enum): values=itertypes; default="slice"
    class axis (attribute): default=0
    class allslice (readonly):
        "a slice object representing the entire grid"
        def get(_, self):
            try: return self.iterator.allslice
            except AttributeError: return slice(0, self.shape[0])

    # for parcel iterators
    clone(ParcelIterator.parcelmap, readonly=False)
    clone(ParcelIterator.parcelseq, readonly=False)

    # delegates
    deferto(mapping, ("input_coords", "output_coords"))


    @staticmethod
    def from_start_step(names=space, shape=[], start=[], step=[]): 
        """
        Create a SamplingGrid instance from sequences of names, shape, start
        and step.
        """
        indim = []
        outdim = []
        ndim = len(names)
        # fill in default step size
        step = N.array(step)
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
        "@return an identity grid of the given shape."
        ndim = len(shape)
        if len(names) != ndim:
            raise ValueError('shape and number of axisnames do not agree')
        w = Mapping.identity(ndim, names=names)
        return SamplingGrid(shape=list(shape), mapping=w)


    def __init__(self, shape, mapping):
        self.shape = tuple(shape)
        self.mapping = mapping


    def range(self):
        "@return the coordinate values in the same format as numpy.indices."
        indices = N.indices(self.shape)
        tmp_shape = indices.shape
        # reshape indices to be a sequence of coordinates
        indices.shape = (self.mapping.ndim, N.product(self.shape))
        _range = self.mapping(indices)
        _range.shape = tmp_shape
        return _range 


    def __iter__(self):
        itermethod = {
          "slice": self.iterslices,
          "parcel": self.iterparcels,
          "slice/parcel": self.itersliceparcels
        }.get(self.itertype)
        if itermethod is None:
            raise ValueError("unknown itertype %s"%`self.itertype`)
        return itermethod()


    def iterslices(self, axis=None):
        if axis is None: axis = self.axis
        self.iterator = SliceIterator(self.shape, axis=axis)
        return self


    def iterparcels(self, parcelseq=None):
        if parcelseq is None: parcelseq = self.parcelseq
        self.iterator = ParcelIterator(self.parcelmap, parcelseq)
        return self


    def itersliceparcels(self, parcelseq=None):
        if parcelseq is None: parcelseq = self.parcelseq
        self.iterator = SliceParcelIterator(self.parcelmap, parcelseq)
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
        g.itertype = 'slice'
        return iter(g)


    def transform(self, matrix): self.mapping = matrix * self.mapping


    def matlab2python(self):
        return SamplingGrid(shape=reverse(self.shape),
          mapping=self.mapping.matlab2python())


    def python2matlab(self):
        return SamplingGrid(shape=reverse(self.shape),
          mapping=self.mapping.python2matlab())

    def replicate(self, n, concataxis=None):
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

    class grids (readonly):
        implements=tuple
        def set(_, self, grids):

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

            readonly.set(_, self, tuple(grids))

    class shape (readonly):
        def init(_, self): return (len(self.grids),) + self.grids[0].shape

    class mapping (readonly):
        def init(_, self):
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
              '%s:%s'%(ic.name, self.concataxis), [newaxis] + ic.axes)
            oc = self.grids[0].mapping.output_coords
            newout = CoordinateSystem(
              '%s:%s'%(oc.name, self.concataxis), [newaxis] + oc.axes)
            return Mapping(newin, newout, mapfunc)

    class concataxis (readonly): default="concat"


    def __init__(self, grids, concataxis=None):
        self.grids = grids
        if concataxis is not None: self.concataxis = concataxis


    def subgrid(self, i): return self.grids[i]

class ConcatenatedIdenticalGrids(ConcatenatedGrids):


    def __init__(self, grid, n, concataxis=None):
        self.grids = [grid for i in range(n)]
        if concataxis is not None: self.concataxis = concataxis

    class mapping (readonly):
        def init(_, self):

            newaxis = Axis(name=self.concataxis)
            ic = self.grids[0].mapping.input_coords
            newin = CoordinateSystem(
              '%s:%s'%(ic.name, self.concataxis), [newaxis] + ic.axes)
            oc = self.grids[0].mapping.output_coords
            newout = CoordinateSystem(
              '%s:%s'%(oc.name, self.concataxis), [newaxis] + oc.axes)

            t = self.grids[0].mapping.transform
            ndim = t.shape[0]-1
            T = N.zeros((ndim+2,)*2, N.float64)
            T[0:ndim,0:ndim] = t[0:ndim,0:ndim]
            T[0:ndim,-1] = t[0:ndim,-1]
            T[ndim,ndim] = 1.
            T[(ndim+1),(ndim+1)] = 1.
            return Affine(newin, newout, T)

class SliceGrid(SamplingGrid):
    """
    Return an affine slice of a given grid with specified
    origin, steps and shape.
    """

    def __init__(self, grid, origin, directions, shape):
        self.fmatrix = N.zeros((self.nout, self.ndim), N.float64)
        _axes = []
        for i in range(directions.shape[0]):
            self.fmatrix[i] = directions[i]
            _axes.append(axis.VoxelAxis(len=shape[i], name=axis.space[i]))
        in_coords = coordinate_system.CoordinateSystem('voxel', _axes)
        self.fvector = origin
        mapping = DegenerateAffine(in_coords, output_coords, fmatrix, fvector)
        SamplingGrid.__init__(self, shape=shape, mapping=mapping)
