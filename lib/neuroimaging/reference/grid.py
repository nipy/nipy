import enthought.traits as traits
import numpy as N

from attributes import readonly

from neuroimaging.reference.mapping import Mapping, Affine, DegenerateAffine
from neuroimaging.reference.axis import space, RegularAxis, VoxelAxis, Axis
from neuroimaging.reference.coordinate_system import VoxelCoordinateSystem,\
  DiagonalCoordinateSystem, CoordinateSystem
from neuroimaging.reference.grid_iterators import SliceIterator, ParcelIterator,\
  SliceParcelIterator, AllSliceIterator


##############################################################################
class SamplingGrid(traits.HasTraits):

    mapping = traits.Any()
    shape = traits.ListInt()
    itertype = traits.Trait('slice', 'parcel', 'slice/parcel', 'all')

    # for parcel iterators
    labels = traits.Any()
    labelset = traits.Any()

    # for slice iterators
    end = traits.Any()
    start = traits.Any()
    step = traits.Any()
    axis = traits.Int(0)

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
        "Return the identity SamplingGrid based on a given shape."
        ndim = len(shape)
        if len(names) != ndim:
            raise ValueError('shape and number of axisnames do not agree')
        w = Mapping.identity(ndim, names=names)
        return SamplingGrid(shape=list(shape), mapping=w)

    #-------------------------------------------------------------------------
    def __init__(self, shape, mapping):
        traits.HasTraits.__init__(self, shape=shape, mapping=mapping)

    #-------------------------------------------------------------------------
    def range(self):
        "Return the coordinate values of a SamplingGrid."
        tmp = N.indices(self.shape)
        tmp_shape = tmp.shape
        tmp.shape = (self.mapping.ndim, N.product(self.shape))
        tmp = self.mapping(tmp)
        tmp.shape = tmp_shape
        return tmp 

    #-------------------------------------------------------------------------
    def __iter__(self):
        if self.itertype is 'slice':
            if self.end is None:
                self.end = self.shape
            if self.start is None:
                self.start = [0] * len(self.shape)
            if self.step is None:
                self.step = [1] * len(self.shape)
            self.iterator = iter(SliceIterator(self.end,
                                               start=self.start,
                                               step=self.step,
                                               axis=self.axis))
        elif self.itertype is 'all':
            self.iterator = iter(AllSliceIterator(self.shape))
        elif self.itertype is 'parcel':
            self.iterator = iter(ParcelIterator(self.labels,
                                                    self.labelset))
        elif self.itertype is 'slice/parcel':
            self.iterator = iter(SliceParcelIterator(self.labels,
                                                     self.labelset))
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
            ndim = len(self.shape)
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
        return SamplingGrid(shape=self.shape[::-1],
          mapping=self.mapping.matlab2python())

    #-------------------------------------------------------------------------
    def python2matlab(self):
        return SamplingGrid(shape=self.shape[::-1],
          mapping=self.mapping.python2matlab())


###############################################################################
class ConcatenatedGrids(SamplingGrid):
    """
    Return a grid formed by concatenating a sequence of grids. Checks are done
    to ensure that the coordinate systems are consistent, as is the shape.

    It returns a grid with the proper shape but no inverse.

    This is most likely the kind of grid to be used for fMRI images.
    """

    grids = traits.List()
    concataxis = traits.Str('concat')

    #-------------------------------------------------------------------------
    def __init__(self, grids, **keywords):
        traits.HasTraits.__init__(self, grids=grids, **keywords)
        SamplingGrid.__init__(self, shape=self.shape, mapping=self.mapping)

    #-------------------------------------------------------------------------
    def _grids_changed(self):
        n = len(self.grids)
        self.shape = [n] + self.grids[0].shape

        # check mappings are affine
        check = N.sum([not isinstance(self.grids[i].mapping, Affine)\
                      for i in range(n)])
        if check: raise ValueError('must all be affine mappings!')

        # check shapes are identical
        s = self.grids[0].shape
        check = N.sum([self.grids[i].shape != s for i in range(n)])
        if check: raise ValueError(
          'shape must be the same in ConcatenatedGrids')

        # check input coordinate systems are identical
    
        ic = self.grids[0].mapping.input_coords
        check = N.sum([self.grids[i].mapping.input_coords != ic for i in range(n)])
        if check: raise ValueError(
          'input coordinate systems must be the same in ConcatenatedGrids')

        # check output coordinate systems are identical
    
        oc = self.grids[0].mapping.output_coords
        check = N.sum([self.grids[i].mapping.output_coords != oc for i in range(n)])
        if check: raise ValueError(
          'output coordinate systems must be the same in concatenate_grids')

        def _mapping(x):
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
        newin = CoordinateSystem('%s:%s' % (ic.name, self.concataxis),
                                 [newaxis] + ic.axes)
        newout = CoordinateSystem('%s:%s' % (oc.name, self.concataxis),
                                 [newaxis] + oc.axes)

        self.mapping = Mapping(newin, newout, _mapping)

    #-------------------------------------------------------------------------
    def subgrid(self, i): return self.grids[i]


##############################################################################
class DuplicatedGrids(ConcatenatedGrids):
    """
    Take a given SamplingGrid and duplicate it j times, returning a
    SamplingGrid with shape=(j,)+grid.shape.
    """
    step = traits.Float(1.)
    start = traits.Float(0.)

    #-------------------------------------------------------------------------
    def __init__(self, grid, j, **keywords):
        ConcatenatedGrids.__init__(self, [grid]*j, **keywords)

    #-------------------------------------------------------------------------
    def _grids_changed(self):
        ConcatenatedGrids._grids_changed(self)
        ndim = len(self.shape)
        t = N.zeros((ndim + 1,)*2, N.Float)
        t[0:(ndim-1),0:(ndim-1)] = self.grids[0].mapping.transform[0:(ndim-1),0:(ndim-1)]
        t[0:(ndim-1),ndim] = self.grids[0].mapping.transform[0:(ndim-1),(ndim-1)]
        t[(ndim-1),(ndim-1)] = self.step
        t[(ndim-1),ndim] = self.start
        t[ndim,ndim] = 1.
        w = Affine(self.mapping.input_coords, self.mapping.output_coords, t)

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
        SamplingGrid.__init__(self, mapping=mapping, shape=shape)
