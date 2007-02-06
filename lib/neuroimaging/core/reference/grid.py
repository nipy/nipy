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


    def copy(self):
        """
        Create a copy of the grid.

        @rtype: L{SamplingGrid}
        """
        return SamplingGrid(self.shape, self.mapping, self.input_coords,
                            self.output_coords)

    def allslice (self):
        """
        A slice object representing the entire grid.
        """
        return slice(0, self.shape[0])

    def range(self):
        """
        Return the coordinate values in the same format as numpy.indices.
        """
        indices = N.indices(self.shape)
        tmp_shape = indices.shape
        # reshape indices to be a sequence of coordinates
        indices.shape = (self.ndim, N.product(self.shape))
        _range = self.mapping(indices)
        _range.shape = tmp_shape
        return _range 


    def slab(self, start, step, count):
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
        return samp_grid


    def transform(self, mapping): 
        """
        Apply a transformation (mapping) to this grid.
        """
        self.mapping = mapping * self.mapping

    def matlab2python(self):
        """
        Convert a grid in matlab-ordered voxels to python ordered voxels.
        See Mapping.matlab2python for more details.
        """
        mapping = self.mapping.matlab2python()
        return SamplingGrid(reverse(self.shape), mapping, 
          self.input_coords.reverse(), self.output_coords)

    def python2matlab(self):
        """
        Convert a grid in python ordered voxels to matlab ordered voxels.
        See Mapping.python2matlab for more details.
        """
        mapping = self.mapping.python2matlab()
        return SamplingGrid(reverse(self.shape), mapping, 
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


    def __init__(self, grids, concataxis="concat"):
        """
        @param grid: The grids to be used.
        @type grid: [L{SamplingGrid}]
        @param concataxis: The name of the new dimension formed by
            concatentation
        @type concataxis: C{str}
        """        
        self.grids = self._grids(grids)
        self.concataxis = concataxis
        mapping, input_coords, output_coords = self._mapping()
        shape = (len(self.grids),) + self.grids[0].shape
        SamplingGrid.__init__(self, shape, mapping, input_coords, output_coords)


    def _grids(self, grids):
        """
        Setup the grids.
        """
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
        """
        Set up the mapping and coordinate systems.
        """
        def mapfunc(x):
            try:
                I = x[0].view(N.int32)
                X = x[1:]
                v = N.zeros(x.shape[1:])
                for j in I.shape[0]:
                    v[j] = self.grids[I[j]].mapping(X[j])
                return v
            except:
                i = int(x[0])
                x = x[1:]
                return self.grids[i].mapping(x)
                
        newaxis = Axis(name=self.concataxis)
        in_coords = self.grids[0].input_coords
        newin = CoordinateSystem('%s:%s'%(in_coords.name, self.concataxis), \
                                 [newaxis] + list(in_coords.axes()))
        out_coords = self.grids[0].output_coords
        newout = CoordinateSystem('%s:%s'%(out_coords.name, self.concataxis), \
                                  [newaxis] + list(out_coords.axes()))
        return Mapping(mapfunc), newin, newout


    def subgrid(self, i):
        """
        Return the i'th grid from the sequence of grids.

        @param i: The grid to return
        @type i: C{int}
        @raise C{IndexError}: if i in out of range.
        """
        return self.grids[i]

class ConcatenatedIdenticalGrids(ConcatenatedGrids):
    """
    A set of concatenated grids, which are all identical.
    """
    
    def __init__(self, grid, n, concataxis="concat"):
        """
        @param grid: The grid to be used.
        @type grid: L{SamplingGrid}
        @param n: The number of times to concatenate the grid
        @type n: C{int}
        @param concataxis: The name of the new dimension formed by
            concatentation
        @type concataxis: C{str}
        """
        ConcatenatedGrids.__init__(self, [grid]*n , concataxis)

    def _mapping(self):
        """
        Set up the mapping and coordinate systems.
        """
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
        out_trans = N.zeros((ndim+2,)*2)
        out_trans[0:ndim, 0:ndim] = in_trans[0:ndim, 0:ndim]
        out_trans[0:ndim, -1] = in_trans[0:ndim, -1]
        out_trans[ndim, ndim] = 1.
        out_trans[(ndim+1), (ndim+1)] = 1.
        return Affine(out_trans), newin, newout

