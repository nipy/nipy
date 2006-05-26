import numpy as N

from attributes import attribute, readonly, deferto, clone
from protocols import Iterator, Sequence

from neuroimaging import reverse
from neuroimaging.reference.mapping import Mapping, Affine, DegenerateAffine
from neuroimaging.reference.axis import space, RegularAxis, VoxelAxis, Axis
from neuroimaging.reference.coordinate_system import VoxelCoordinateSystem,\
  DiagonalCoordinateSystem, CoordinateSystem
from neuroimaging.reference.iterators import itertypes, SliceIterator,\
  ParcelIterator, SliceParcelIterator, AllSliceIterator


##############################################################################
class SamplingGrid (object):

    class shape (readonly): implements=tuple #traits.ListInt()
    class ndim (readonly): get=lambda _, self: len(self.shape)
    class mapping (attribute): implements=Mapping
    class iterator (attribute): implements=Iterator
    class itertype (attribute):
        default="slice"
        def set(_, self, value):
            if value not in itertypes: raise ValueError(
              "itertype must be one of %s"%itertypes)
            attribute.set(_, self, value)
    class axis (attribute): default=0

    # for parcel iterators
    clone(ParcelIterator.parcelmap, readonly=False)
    clone(ParcelIterator.parcelseq, readonly=False)

    # delegates
    deferto(mapping, ("input_coords", "output_coords"))

    #-------------------------------------------------------------------------
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

    #-------------------------------------------------------------------------
    @staticmethod
    def identity(shape=(), names=space):
        "Return an identity grid of the given shape."
        ndim = len(shape)
        if len(names) != ndim:
            raise ValueError('shape and number of axisnames do not agree')
        w = Mapping.identity(ndim, names=names)
        return SamplingGrid(shape=list(shape), mapping=w)

    #-------------------------------------------------------------------------
    def __init__(self, shape, mapping):
        self.shape = tuple(shape)
        self.mapping = mapping

    #-------------------------------------------------------------------------
    def range(self):
        "Return the coordinate values in the same format as numpy.indices."
        indices = N.indices(self.shape)
        tmp_shape = indices.shape
        # reshape indices to be a sequence of coordinates
        indices.shape = (self.mapping.ndim, N.product(self.shape))
        _range = self.mapping(indices)
        _range.shape = tmp_shape
        return _range 

    #-------------------------------------------------------------------------
    def __iter__(self):
        if self.itertype is "all": return self.iterall()
        elif self.itertype is "slice": return self.iterslices()
        elif self.itertype is "parcel": return self.iterparcels()
        elif self.itertype is "slice/parcel": return self.itersliceparcels()

    #-------------------------------------------------------------------------
    def iterall(self):
        self.iterator = iter(AllSliceIterator(self.shape))
        return self

    #-------------------------------------------------------------------------
    def iterslices(self, axis=None):
        if axis is None: axis = self.axis
        self.iterator = iter(SliceIterator(self.shape, axis=self.axis))
        return self

    #-------------------------------------------------------------------------
    def iterparcels(self, parcelseq=None):
        if parcelseq is None: parcelseq = self.parcelseq
        self.iterator = iter(ParcelIterator(self.parcelmap, parcelseq))
        return self

    #-------------------------------------------------------------------------
    def itersliceparcels(self):
        self.iterator = iter(SliceParcelIterator(self.parcelmap, self.parcelseq))
        return self

    #-------------------------------------------------------------------------
    def next(self):
        self.itervalue = self.iterator.next()
        return self.itervalue

    #-------------------------------------------------------------------------
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

    #-------------------------------------------------------------------------
    def transform(self, matrix): self.mapping = matrix * self.mapping

    #-------------------------------------------------------------------------
    def matlab2python(self):
        return SamplingGrid(shape=reverse(self.shape),
          mapping=self.mapping.matlab2python())

    #-------------------------------------------------------------------------
    def python2matlab(self):
        return SamplingGrid(shape=reverse(self.shape),
          mapping=self.mapping.python2matlab())

    #-------------------------------------------------------------------------
    def replicate(self, n):
        """
        Duplicate self n times, returning a ConcatenatedGrids with
        shape == (n,)+self.shape.
        """
        return ConcatenatedGrids([self]*n)



###############################################################################
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
                    I = x[0].view(N.Int)
                    X = x[1:]
                    v = N.zeros(x.shape[1:], N.Float)
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

    #-------------------------------------------------------------------------
    def __init__(self, grids, concataxis=None):
        self.grids = grids
        if concataxis is not None: self.concataxis = concataxis

    #-------------------------------------------------------------------------
    def subgrid(self, i): return self.grids[i]



##############################################################################
class SliceGrid(SamplingGrid):
    """
    Return an affine slice of a given grid with specified
    origin, steps and shape.
    """

    #-------------------------------------------------------------------------
    def __init__(self, grid, origin, directions, shape):
        self.fmatrix = N.zeros((self.nout, self.ndim), N.Float)
        _axes = []
        for i in range(directions.shape[0]):
            self.fmatrix[i] = directions[i]
            _axes.append(axis.VoxelAxis(len=shape[i], name=axis.space[i]))
        input_coords = coordinate_system.CoordinateSystem('voxel', _axes)
        self.fvector = origin
        mapping = DegenerateAffine(
          input_coords, output_coords, fmatrix, fvector)
        SamplingGrid.__init__(self, shape=shape, mapping=mapping)
