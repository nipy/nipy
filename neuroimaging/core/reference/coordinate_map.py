"""
Coordinate maps store all the details about how an image translates to space.
They also provide mechanisms for iterating over that space.
"""
import copy

import numpy as np

from neuroimaging.core.reference.axis import RegularAxis, Axis, VoxelAxis
from neuroimaging.core.reference.coordinate_system import \
  VoxelCoordinateSystem, DiagonalCoordinateSystem, CoordinateSystem
from neuroimaging.core.reference.mapping import Mapping, Affine


__docformat__ = 'restructuredtext'



class CoordinateMap(object):
    """
    Defines a set of input and output coordinate systems and a mapping
    between the two, which represents the mapping of (for example) an
    image from voxel space to real space.
    """
    
    @staticmethod
    def from_start_step(names, start, step, shape):
        """
        Create a `CoordinateMap` instance from sequences of names, shape, start
        and step.

        :Parameters:
            names : ``tuple`` of ``string``
                TODO
            start : ``tuple`` of ``float``
                TODO
            step : ``tuple`` of ``float``
                TODO
            shape: ''tuple'' of ''int''

        :Returns: `CoordinateMap`
        
        :Predcondition: ``len(names) == len(shape) == len(start) == len(step)``
        """
        ndim = len(names)
        # fill in default step size
        step = np.asarray(step)
        axes = [RegularAxis(name=names[i], length=shape[i],
          start=start[i], step=step[i]) for i in range(ndim)]
        input_coords = VoxelCoordinateSystem('voxel', axes)
        output_coords = DiagonalCoordinateSystem('world', axes)
        transform = output_coords.transform()
        
        mapping = Affine(transform)
        return CoordinateMap(mapping, input_coords, output_coords)


    @staticmethod
    def identity(names, shape):
        """
        Return an identity comap of the given shape.
        
        :Parameters:
            shape : ``tuple`` of ``int``
                TODO
            names : ``tuple`` of ``string``
                TODO

        :Returns: `CoordinateMap` with `VoxelCoordinateSystem` input
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
        return CoordinateMap(aff_ident, input_coords, output_coords)

    @staticmethod
    def from_affine(mapping, names, shape):
        """
        Return comap using a given `Affine` mapping
        
        :Parameters:
            mapping : `Affine`
                An affine mapping between the input and output coordinate systems.
            names : ``tuple`` of ``string``
                The names of the axes of the coordinate systems
            shape : ''tuple'' of ''int''
                The shape of the comap
        :Returns: `CoordinateMap`
        
        :Precondition: ``len(shape) == len(names)``
        
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        ndim = len(names)
        if mapping.ndim != ndim:
            raise ValueError('shape and number of axis names do not agree')
        axes = [VoxelAxis(name, length=l) for name, l in zip(names, shape)]
        input_coords = VoxelCoordinateSystem("voxel", axes)
        output_coords = DiagonalCoordinateSystem('world', axes)
        return CoordinateMap(Affine(mapping.transform), input_coords, output_coords)

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
        # These guys define the structure of the comap.
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
        Create a copy of the comap.

        :Returns: `CoordinateMap`
        """
        return CoordinateMap(self.mapping, self.input_coords,
                            self.output_coords)

    def __getitem__(self, index):
        """
        If all input coordinates are VoxelCoordinateSystem, return
        a slice through the comap.

        Parameters
        ----------
        index : ``int`` or ``slice``
            sequence of integers or slices
        
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
            return CoordinateMap(mapping, newic, self.output_coords)
        else:
            raise ValueError, 'input_coords must be VoxelCoordinateSystem for slice of comap to make sense'

    def range(self):
        """
        Return the coordinate values in the same format as numpy.indices.
        
        :Returns: TODO
        """
        if hasattr(self, 'shape'):
            indices = np.indices(self.shape)
            tmp_shape = indices.shape
            # reshape indices to be a sequence of coordinates
            indices.shape = (self.ndim[0], np.product(self.shape))
            _range = self.mapping(indices)
            _range.shape = tmp_shape
            return _range 
        else:
            raise AttributeError, 'range of comap only makes sense if input_coords are VoxelCoordinateSystem'

    def transform(self, mapping): 
        """        
        Apply a transformation (mapping) to this comap.
        
        :Parameters:
            mapping : `mapping.Mapping`
                The mapping to be applied.
        
        :Returns: ``None``
        """
        self.mapping = mapping * self.mapping

    def matlab2python(self):
        """
        Convert a comap in matlab-ordered voxels to python ordered voxels
        if input_coords is an instance of VoxelCoordinateSystem.
        See `Mapping.matlab2python` for more details.


        """
        if isinstance(self.input_coords, VoxelCoordinateSystem):
            mapping = self.mapping.matlab2python()
            newi = self.input_coords.reverse()
            return CoordinateMap(mapping, 
                                VoxelCoordinateSystem(newi.name, newi.axes()),
                                self.output_coords.reverse())
        else:
            raise ValueError, 'input_coords must be VoxelCoordinateSystem for self.matlab2python to make sense'

    def python2matlab(self):
        """
        Convert a comap in python ordered voxels to matlab ordered voxels
        if input_coords is an instance of VoxelCoordinateSystem.
        See `Mapping.python2matlab` for more details.
        """
        if isinstance(self.input_coords, VoxelCoordinateSystem):
            mapping = self.mapping.python2matlab()
            newi = self.input_coords.reverse()
            return CoordinateMap(mapping, 
                                VoxelCoordinateSystem(newi.name, newi.axes()),
                                self.output_coords.reverse())
        else:
            raise ValueError, 'input_coords must be VoxelCoordinateSystem for self.python2matlab to make sense'

    def replicate(self, n, concataxis="concat"):
        """
        Duplicate self n times, returning a `ConcatenatedComaps` with
        shape == (n,)+self.shape.
        
        :Parameters:
            n : ``int``
                TODO
            concataxis : ``string``
                The name of the new dimension formed by concatenation
        """
        return ConcatenatedIdenticalComaps(self, n, concataxis=concataxis)

def centered_comap(shape, pixdims=(1,1,1),names=('zdim','ydim', 'xdim')):
    """
    creates a simple centered comap that centers matrix on zero

    If you have a nd-array and just want a simple comap that puts the center of
    your data matrix at approx (0,0,0)...this will generate the comap you need

    Parameters
    _________
    shape   : tuple
        shape of the data matrix (90, 109, 90)
    pixdims : tuple
        tuple if ints maps voxel to real-world  mm size (2,2,2)
    names   : tuple 
        tuple of names describing axis ('zaxis', 'yaxis', 'xaxis')

    :Returns: `CoordinateMap`
        
        :Predcondition: ``len(shape) == len(pixdims) == len(names)``

    Put in a catch for ndims > 3 as time vectors are rarely start < 0
    """
    if not len(shape) == len(pixdims) == len(names):
        print 'Error: len(shape) == len(pixdims) == len(names)'
        return None
    ndim = len(names)
    # fill in default step size
    step = np.asarray(pixdims)
    ashape = np.asarray(shape)
    # start = 
    start = ashape * np.abs(step) /2 * np.sign(step)*-1
    axes = [RegularAxis(name=names[i], length=ashape[i],
                        start=start[i], step=step[i]) for i in range(ndim)]
    input_coords = VoxelCoordinateSystem('voxel', axes)
    output_coords = DiagonalCoordinateSystem('world', axes)
    transform = output_coords.transform()
        
    mapping = Affine(transform)
    return CoordinateMap(mapping, input_coords, output_coords)
   
        
        
    

class ConcatenatedComaps(CoordinateMap):
    """
    Return a comap formed by concatenating a sequence of comaps. Checks are done
    to ensure that the coordinate systems are consistent, as is the shape.
    It returns a comap with the proper shape but no inverse.
    This is most likely the kind of comap to be used for fMRI images.
    """


    def __init__(self, comaps, concataxis="concat"):
        """
        :Parameters:
            comap : ``[`CoordinateMap`]``
                The comaps to be used.
            concataxis : ``string``
                The name of the new dimension formed by concatenation
        """        
        self.comaps = self._comaps(comaps)
        self.concataxis = concataxis
        mapping, input_coords, output_coords = self._mapping()
        CoordinateMap.__init__(self, mapping, input_coords, output_coords)


    def _getshape(self):
        return (len(self.comaps),) + self.comaps[0].shape
    shape = property(_getshape)

    def _comaps(self, comaps):
        """
        Setup the comaps.
        """
        # check mappings are affine
        check = np.any([not isinstance(comap.mapping, Affine)\
                          for comap in comaps])
        if check:
            raise ValueError('must all be affine mappings!')

        # check shapes are identical
        s = comaps[0].shape
        check = np.any([comap.shape != s for comap in comaps])
        if check:
            raise ValueError('subcomaps must have same shape')

        # check input coordinate systems are identical
        in_coords = comaps[0].input_coords
        check = np.any([comap.input_coords != in_coords\
                           for comap in comaps])
        if check:
            raise ValueError(
              'subcomaps must have same input coordinate systems')

        # check output coordinate systems are identical
        out_coords = comaps[0].output_coords
        check = np.any([comap.output_coords != out_coords\
                           for comap in comaps])
        if check:
            raise ValueError(
              'subcomaps must have same output coordinate systems')
        return tuple(comaps)

    def _mapping(self):
        """
        Set up the mapping and coordinate systems.
        """
        def mapfunc(x):
            try:
                I = x[0].view(np.int32)
                X = x[1:]
                v = np.zeros(x.shape[1:])
                for j in I.shape[0]:
                    v[j] = self.comaps[I[j]].mapping(X[j])
                return v
            except:
                i = int(x[0])
                x = x[1:]
                return self.comaps[i].mapping(x)
                
        newaxis = Axis(name=self.concataxis)
        in_coords = self.comaps[0].input_coords
        newin = CoordinateSystem('%s:%s'%(in_coords.name, self.concataxis), \
                                 [newaxis] + list(in_coords.axes()))
        out_coords = self.comaps[0].output_coords
        newout = CoordinateSystem('%s:%s'%(out_coords.name, self.concataxis), \
                                  [newaxis] + list(out_coords.axes()))
        return Mapping(mapfunc), newin, newout


    def subcomap(self, i):
        """
        Return the i'th comap from the sequence of comaps.

        :Parameters:
           i : ``int``
               The index of the comap to return

        :Returns: `CoordinateMap`
        
        :Raises IndexError: if i in out of range.
        """
        return self.comaps[i]

class ConcatenatedIdenticalComaps(ConcatenatedComaps):
    """
    A set of concatenated comaps, which are all identical.
    """
    
    def __init__(self, comap, n, concataxis="concat"):
        """
        :Parameters:
            comap : `CoordinateMap`
                The comap to be used
            n : ``int``
                The number of tiems to concatenate the comap
            concataxis : ``string``
                The name of the new dimension formed by concatenation
        """
        ConcatenatedComaps.__init__(self, [comap]*n , concataxis)

    def _mapping(self):
        """
        Set up the mapping and coordinate systems.
        """
        newaxis = Axis(name=self.concataxis)
        in_coords = self.comaps[0].input_coords
        newin = CoordinateSystem(
            '%s:%s'%(in_coords.name, self.concataxis), \
               [newaxis] + list(in_coords.axes()))
        out_coords = self.comaps[0].output_coords
        newout = CoordinateSystem(
            '%s:%s'%(out_coords.name, self.concataxis), \
               [newaxis] + list(out_coords.axes()))

        in_trans = self.comaps[0].mapping.transform
        ndim = in_trans.shape[0]-1
        out_trans = np.zeros((ndim+2,)*2)
        out_trans[0:ndim, 0:ndim] = in_trans[0:ndim, 0:ndim]
        out_trans[0:ndim, -1] = in_trans[0:ndim, -1]
        out_trans[ndim, ndim] = 1.
        out_trans[(ndim+1), (ndim+1)] = 1.
        return Affine(out_trans), newin, newout

