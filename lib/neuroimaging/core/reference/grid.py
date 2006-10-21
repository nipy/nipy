"""
Samplng grids store all the details about how an image translates to space.
They also provide mechanisms for iterating over that space.
"""
import numpy as N

from neuroimaging import reverse
from neuroimaging.core.reference.mapping import Mapping, Affine
from neuroimaging.core.reference.axis import space, RegularAxis, Axis, VoxelAxis
from neuroimaging.core.reference.coordinate_system import \
  VoxelCoordinateSystem, DiagonalCoordinateSystem, CoordinateSystem
from neuroimaging.core.reference.iterators import SliceIterator, \
  ParcelIterator, SliceParcelIterator


class SamplingGrid (object):
    """
    Defines a set of input and output coordinate systems and a mapping between the
    two, which represents the mapping of (for example) an image from voxel space
    to real space.
    """
    
    @staticmethod
    def from_start_step(names=space, shape=(), start=(), step=()): 
        """
        Create a SamplingGrid instance from sequences of names, shape, start
        and step.
        """
        ndim = len(names)
        # fill in default step size
        step = N.asarray(step)
        axes = [RegularAxis(name=names[i], length=shape[i],
          start=start[i], step=step[i]) for i in range(ndim)]
        input_coords = VoxelCoordinateSystem('voxel', axes)
        output_coords = DiagonalCoordinateSystem('world', axes)
        transform = output_coords.transform()
        
        mapping = Affine(transform)
        return SamplingGrid(list(shape), mapping, input_coords, output_coords)


    @staticmethod
    def identity(shape=(), names=space):
        """
        return an identity grid of the given shape.
        """
        ndim = len(shape)
        if len(names) != ndim:
            raise ValueError('shape and number of axis names do not agree')
        axes = [VoxelAxis(name) for name in names]

        input_coords = VoxelCoordinateSystem('voxel', axes)
        output_coords = DiagonalCoordinateSystem('world', axes)
        aff_ident = Affine.identity(ndim)
        return SamplingGrid(list(shape), aff_ident, input_coords, output_coords)

    @staticmethod
    def from_affine(mapping, shape=(), names=space):
        """
        Return grid using a given affine mapping
        """
        ndim = len(names)
        if mapping.ndim() != ndim:
            raise ValueError('shape and number of axis names do not agree')
        axes = [VoxelAxis(name) for name in names]

        input_coords = VoxelCoordinateSystem('voxel', axes)
        output_coords = DiagonalCoordinateSystem('world', axes)

        return SamplingGrid(list(shape), mapping, input_coords, output_coords)



    def __init__(self, shape, mapping, input_coords, output_coords):
        # These guys define the structure of the grid.
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.mapping = mapping
        self.input_coords = input_coords
        self.output_coords = output_coords

        # These guys are for use of the SamplingGrid as an iterator.
        iterators = {"slice": (SliceIterator, ["shape", "axis"]),
                     "parcel": (ParcelIterator, ["parcelmap", "parcelseq"]),
                     "slice/parcel": (SliceParcelIterator, ["parcelmap", "parcelseq"])}
        self._iterguy = \
          self._IterHelper(self.shape, 0, "slice", None, None, iterators)

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
        indices.shape = (self.ndim, N.product(self.shape))
        _range = self.mapping(indices)
        _range.shape = tmp_shape
        return _range 

    def __iter__(self):
        iter(self._iterguy)
        return self
        
    def next(self):
        return self._iterguy.next()
    
    def itervalue(self):
        """
        Return the current iteration value
        """
        return self._iterguy.itervalue
                
    def set_iter_param(self, name, val):
        """ Set an iteration parameter """
        self._iterguy.set(name, val)
        
    def get_iter_param(self, name):
        """ Get an iteration parameter """
        return self._iterguy.get(name)
    
    class _IterHelper:
        """
        This class takes care of all the seedy details of iteration
        which should be sufficiently hidden from the outside world.
        """
        def __init__(self, shape, axis, itertype, parcelseq, parcelmap, iterators):
            self.dict = {"shape": shape,
                         "axis": axis,
                         "itertype": itertype,
                         "parcelseq": parcelseq,
                         "parcelmap": parcelmap}
            self.iterators = iterators

        def set(self, name, val):
            if name not in self.dict.keys():
                raise KeyError
            self.dict[name] = val
        
        def get(self, name):
            return self.dict[name]

        def __iter__(self):
            itertype = self.dict["itertype"]
            iterator, params = self.iterators[itertype]
            self.iterator = iterator(*[self.dict[key] for key in params])
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
            trans = self.mapping.transform.copy()
            trans[0:ndim, ndim] = self.mapping(start)
            trans[ndim, ndim] = 1.
            for i in range(ndim):
                v = N.zeros((ndim,))
                w = v.copy()
                v[i] = step[i]
                trans[0:ndim, i] = self.mapping(v) - self.mapping(w)
            _map = Affine(trans)
        else:
            def __map(x, start=start, step=step, _f=self.mapping):
                v = start + step * x
                return _f(v)
            _map = Mapping(__map)

        samp_grid = \
          SamplingGrid(count, _map, self.input_coords, self.output_coords)
        samp_grid.set_iter_param("axis", axis)
        return iter(samp_grid)


    def transform(self, mapping): 
        self.mapping = mapping * self.mapping

    def matlab2python(self):
        m = self.mapping.matlab2python()
        return SamplingGrid(reverse(self.shape), m, 
          self.input_coords.reverse(), self.output_coords)

    def python2matlab(self):
        m = self.mapping.python2matlab()
        return SamplingGrid(reverse(self.shape), m, 
          self.input_coords.reverse(), self.output_coords)


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
        check = N.any([not isinstance(grid.mapping, Affine)\
                          for grid in grids])
        if check:
            raise ValueError('must all be affine mappings!')

        # check shapes are identical
        s = grids[0].shape
        check = N.any([grid.shape != s for grid in grids])
        if check:
            raise ValueError('subgrids must have same shape')

        # check input coordinate systems are identical
        in_coords = grids[0].input_coords
        check = N.any([grid.input_coords != in_coords\
                           for grid in grids])
        if check:
            raise ValueError(
              'subgrids must have same input coordinate systems')

        # check output coordinate systems are identical
        out_coords = grids[0].output_coords
        check = N.any([grid.output_coords != out_coords\
                           for grid in grids])
        if check:
            raise ValueError(
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
        in_coords = self.grids[0].input_coords
        newin = CoordinateSystem(
              '%s:%s'%(in_coords.name, self.concataxis), \
                 [newaxis] + list(in_coords.axes()))
        out_coords = self.grids[0].output_coords
        newout = CoordinateSystem(
              '%s:%s'%(out_coords.name, self.concataxis), \
                 [newaxis] + list(out_coords.axes()))
        return Mapping(mapfunc), newin, newout


    def __init__(self, grids, concataxis="concat"):
        self.grids = self._grids(grids)
        self.concataxis = concataxis
        mapping, input_coords, output_coords = self._mapping()
        shape = (len(self.grids),) + self.grids[0].shape
        SamplingGrid.__init__(self, shape, mapping, input_coords, output_coords)

    def subgrid(self, i): 
        return self.grids[i]

class ConcatenatedIdenticalGrids(ConcatenatedGrids):

    def __init__(self, grid, n, concataxis="concat"):
        ConcatenatedGrids.__init__(self, [grid]*n , concataxis)
        self.mapping, self.input_coords, self.output_coords = self._mapping()

    def _mapping(self):
        newaxis = Axis(name=self.concataxis)
        in_coords = self.grids[0].input_coords
        newin = CoordinateSystem(
            '%s:%s'%(in_coords.name, self.concataxis), \
               [newaxis] + list(in_coords.axes()))
        out_coords = self.grids[0].output_coords
        newout = CoordinateSystem(
            '%s:%s'%(out_coords.name, self.concataxis), \
               [newaxis] + list(out_coords.axes()))

        in_trans = self.grids[0].mapping.transform
        ndim = in_trans.shape[0]-1
        out_trans = N.zeros((ndim+2,)*2, N.float64)
        out_trans[0:ndim, 0:ndim] = in_trans[0:ndim, 0:ndim]
        out_trans[0:ndim, -1] = in_trans[0:ndim, -1]
        out_trans[ndim, ndim] = 1.
        out_trans[(ndim+1), (ndim+1)] = 1.
        return Affine(out_trans), newin, newout

