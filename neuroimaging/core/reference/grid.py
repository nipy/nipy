"""
Samplng grids store all the details about how an image translates to space.
They also provide mechanisms for iterating over that space.
"""

__docformat__ = 'restructuredtext'

import copy
import numpy as N

from coordinate_system import _reverse
from neuroimaging.core.reference.mapping import Mapping, Affine
from neuroimaging.core.reference.axis import space, RegularAxis, Axis, VoxelAxis
from neuroimaging.core.reference.coordinate_system import \
  VoxelCoordinateSystem, DiagonalCoordinateSystem, CoordinateSystem

class SamplingGrid(object):
    """
    Defines a set of input and output coordinate systems and a mapping between the
    two, which represents the mapping of (for example) an image from voxel space
    to real space.
    """
    
    @staticmethod
    def from_start_step(names=space, shape=(), start=(), step=()): 
        """
        Create a `SamplingGrid` instance from sequences of names, shape, start
        and step.

        :Parameters:
            names : ``tuple`` of ``string``
                TODO
            shape : ``tuple`` of ``int``
                TODO
            start : ``tuple`` of ``float``
                TODO
            step : ``tuple`` of ``float``
                TODO

        :Returns: `SamplingGrid`
        
        :Predcondition: ``len(names) == len(shape) == len(start) == len(step)``
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
        return SamplingGrid(mapping, input_coords, output_coords)


    @staticmethod
    def identity(shape=(), names=space):
        """
        Return an identity grid of the given shape.
        
        :Parameters:
            shape : ``tuple`` of ``int``
                TODO
            names : ``tuple`` of ``string``
                TODO

        :Returns: `SamplingGrid` with `VoxelCoordinateSystem` input
                  and an identity transform 
        
        :Precondition: ``len(shape) == len(names)``
        
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        ndim = len(shape)
        if len(names) != ndim:
            raise ValueError('shape and number of axis names do not agree')
        iaxes = [VoxelAxis(name, l) for l, name in zip(shape, names)]
        oaxes = [VoxelAxis(name) for name in names]

        input_coords = VoxelCoordinateSystem('voxel', iaxes)
        output_coords = DiagonalCoordinateSystem('world', oaxes)
        aff_ident = Affine.identity(ndim)
        return SamplingGrid(aff_ident, input_coords, output_coords)

    @staticmethod
    def from_affine(mapping, shape=(), names=space):
        """
        Return grid using a given `Affine` mapping
        
        :Parameters:
            mapping : `Affine`
                An affine mapping between the input and output coordinate systems.
            shape : ``tuple`` of ``int``
                The shape of the grid
            names : ``tuple`` of ``string``
                The names of the axes of the coordinate systems

        :Returns: `SamplingGrid`
        
        :Precondition: ``len(shape) == len(names)``
        
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        ndim = len(names)
        if mapping.ndim() != ndim:
            raise ValueError('shape and number of axis names do not agree')
        axes = [VoxelAxis(name) for name in names]

        if shape:
            input_coords = VoxelCoordinateSystem('voxel', axes)
        else:
            input_coords = DiagonalCoordinateSystem('voxel', axes)
        output_coords = DiagonalCoordinateSystem('world', axes)

        return SamplingGrid(mapping, input_coords, output_coords)

    def __init__(self, mapping, input_coords, output_coords):
        """
        :Parameters:
            mapping : `mapping.Mapping`
                The mapping between input and output coordinates
            input_coords : `CoordinateSystem`
                The input coordinate system
            output_coords : `CoordinateSystem`
                The output coordinate system
        """
        # These guys define the structure of the grid.
        self.mapping = mapping
        self.input_coords = input_coords
        self.output_coords = output_coords

    def _getshape(self):
        if isinstance(self.input_coords, VoxelCoordinateSystem):
            s = tuple([a.length for a in self.input_coords.axes()])
            return s
        else:
            raise AttributeError, "input_coords must be a VoxelCoordinateSystem to have a shape"
    shape = property(_getshape)

    def _getndim(self):
        return (len(self.input_coords.axes()), len(self.output_coords.axes()))
    ndim = property(_getndim)

    def isaffine(self):
        if isinstance(self.mapping, Affine):
            return True
        return False

    def _getaffine(self):
        if hasattr(self.mapping, "transform"):
            return self.mapping.transform
        raise AttributeError
    affine = property(_getaffine)

    def __call__(self, x):
        """
        Return self.mapping(x)
        """
        return self.mapping(x)

    def copy(self):
        """
        Create a copy of the grid.

        :Returns: `SamplingGrid`
        """
        return SamplingGrid(self.mapping, self.input_coords,
                            self.output_coords)

    def allslice(self):
        """
        TODO: where is this still used?
        A slice object representing the entire grid.
        
        :Returns: ``slice``
        """
        return slice(0, self.shape[0])

    def __getitem__(self, index):
        """
        If all input coordinates are VoxelCoordinateSystem, return
        a slice through the grid.

        """

        if isinstance(self.input_coords, VoxelCoordinateSystem):
            varcoords, mapping, shape = self.mapping._slice_mapping(index, self.shape)

            ia = self.input_coords.axes()
            newia = []
            for i in range(self.ndim[0]):
                if i in varcoords:
                    a = copy.deepcopy(ia[i])
                    newia.append(a)
            newic = VoxelCoordinateSystem(self.input_coords.name, newia, shape=shape)
            return SamplingGrid(mapping, newic, self.output_coords)
        else:
            raise ValueError, 'input_coords must be VoxelCoordinateSystem for slice of grid to make sense'

    def range(self):
        """
        Return the coordinate values in the same format as numpy.indices.
        
        :Returns: TODO
        """
        if hasattr(self, 'shape'):
            indices = N.indices(self.shape)
            tmp_shape = indices.shape
            # reshape indices to be a sequence of coordinates
            indices.shape = (self.ndim[0], N.product(self.shape))
            _range = self.mapping(indices)
            _range.shape = tmp_shape
            return _range 
        else:
            raise AttributeError, 'range of grid only makes sense if input_coords are VoxelCoordinateSystem'

    def transform(self, mapping): 
        """        
        Apply a transformation (mapping) to this grid.
        
        :Parameters:
            mapping : `mapping.Mapping`
                The mapping to be applied.
        
        :Returns: ``None``
        """
        self.mapping = mapping * self.mapping

    def matlab2python(self):
        """
        Convert a grid in matlab-ordered voxels to python ordered voxels
        if input_coords is an instance of VoxelCoordinateSystem.
        See `Mapping.matlab2python` for more details.


        """
        if isinstance(self.input_coords, VoxelCoordinateSystem):
            mapping = self.mapping.matlab2python()
            newi = self.input_coords.reverse()
            return SamplingGrid(mapping, 
                                VoxelCoordinateSystem(newi.name, newi.axes()),
                                self.output_coords.reverse())
        else:
            raise ValueError, 'input_coords must be VoxelCoordinateSystem for self.matlab2python to make sense'

    def python2matlab(self):
        """
        Convert a grid in python ordered voxels to matlab ordered voxels
        if input_coords is an instance of VoxelCoordinateSystem.
        See `Mapping.python2matlab` for more details.
        """
        if isinstance(self.input_coords, VoxelCoordinateSystem):
            mapping = self.mapping.python2matlab()
            newi = self.input_coords.reverse()
            return SamplingGrid(mapping, 
                                VoxelCoordinateSystem(newi.name, newi.axes()),
                                self.output_coords.reverse())
        else:
            raise ValueError, 'input_coords must be VoxelCoordinateSystem for self.python2matlab to make sense'

    def replicate(self, n, concataxis="concat"):
        """
        Duplicate self n times, returning a `ConcatenatedGrids` with
        shape == (n,)+self.shape.
        
        :Parameters:
            n : ``int``
                TODO
            concataxis : ``string``
                The name of the new dimension formed by concatenation
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
        :Parameters:
            grid : ``[`SamplingGrid`]``
                The grids to be used.
            concataxis : ``string``
                The name of the new dimension formed by concatenation
        """        
        self.grids = self._grids(grids)
        self.concataxis = concataxis
        mapping, input_coords, output_coords = self._mapping()
        shape = (len(self.grids),) + self.grids[0].shape
        SamplingGrid.__init__(self, mapping, input_coords, output_coords)


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

        :Parameters:
           i : ``int``
               The index of the grid to return

        :Returns: `SamplingGrid`
        
        :Raises IndexError: if i in out of range.
        """
        return self.grids[i]

class ConcatenatedIdenticalGrids(ConcatenatedGrids):
    """
    A set of concatenated grids, which are all identical.
    """
    
    def __init__(self, grid, n, concataxis="concat"):
        """
        :Parameters:
            grid : `SamplingGrid`
                The grid to be used
            n : ``int``
                The number of tiems to concatenate the grid
            concataxis : ``string``
                The name of the new dimension formed by concatenation
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

