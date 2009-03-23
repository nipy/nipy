"""
Some CoordinateMaps have input_coords that are 'array' coordinates,
hence the function of the CoordinateMap can be evaluated at these
'array' points.

This module tries to make these operations easier by defining a class
ArrayCoordMap that is essentially a CoordinateMap and a shape.

This class has two properties: values, transposed_values the
CoordinateMap at np.indices(shape).

The class Grid is meant to take a CoordinateMap and an np.mgrid-like
notation to create an ArrayCoordMap.
"""
import numpy as np
from coordinate_map import CoordinateMap, Affine, compose
from coordinate_map import product as cmap_product
from coordinate_system import CoordinateSystem

class ArrayCoordMap(object):
    """
    When the input_coords of a CoordinateMap can be thought of as
    'array' coordinates, i.e. an 'input_shape' makes sense. We can
    than evaluate the CoordinateMap at np.indices(input_shape)
    """

    def __init__(self, coordmap, shape):
        """
        Parameters
        ----------
        coordmap : ``CoordinateMap``
           A CoordinateMap with input_coords that are 'array'
           coordinates.
        shape : sequence of int
           The size of the (implied) underlying array.
        """
        self.coordmap = coordmap
        self.shape = tuple(shape)

    def _evaluate(self, transpose=False):
        """
        If the coordmap has a shape (so that it can be thought of as a
        map from voxels to some output space), return the range of the
        coordmap, i.e. the value at all the voxels.

        Parameters
        ----------
        coordmap : `CoordinateMap`
        transpose : bool, optional
           If False (the default), the result is a 2-dimensional
           ndarray with shape[1] == coordmap.ndim[1]. That is, the
           result is a list of output values.  Otherwise, the shape is
           (coordmap.ndim[1],) + coordmap.shape.

        Returns
        -------
        values : array
           Values of self.coordmap evaluated at np.indices(self.shape).
        """

        indices = np.indices(self.shape).astype(
            self.coordmap.input_coords.coord_dtype)
        tmp_shape = indices.shape

        # reshape indices to be a sequence of coordinates
        indices.shape = (self.coordmap.ndim[0], np.product(self.shape))
        _range = self.coordmap(indices.T)
        if transpose:
            _range = _range.T
            _range.shape = (_range.shape[0],) + tmp_shape[1:]
        return _range 

    def _getvalues(self):
        return self._evaluate(transpose=False)
    values = property(_getvalues, 
                      doc='Get values of ArrayCoordMap in a '
                      '2-dimensional array of shape '
                      '(product(self.shape), self.coordmap.ndim[1]))')

    def _getindices_values(self):
        return self._evaluate(transpose=True)
    transposed_values = property(_getindices_values, doc='Get values of ArrayCoordMap in an array of shape (self.coordmap.ndim[1],) + self.shape)')

    def __getitem__(self, index):
        """
        Return a slice through the coordmap.

        Parameters
        ----------
        index : ``int`` or ``slice``
            sequence of integers or slices
        
        """

        if type(index) != type(()):
            index = (index,)
        return _slice(self.coordmap, self.shape, *index)

    @staticmethod
    def from_shape(coordmap, shape):
        """
        Create an evaluator assuming that coordmap.input_coords
        are 'array' coordinates.

        """
        slices = tuple([slice(0,s,1) for s in shape])
        return Grid(coordmap)[slices]

def _slice(coordmap, shape, *slices):
    """
    Slice a 'voxel' CoordinateMap's input_coords with slices. A
    'voxel' CoordinateMap is interpreted as a coordmap having a shape.
    """
    
    if len(slices) < coordmap.ndim[0]:
        slices = (list(slices) +
                  [slice(None,None,None)] * (coordmap.ndim[0] - len(slices)))

    ranges = [np.arange(s) for s in shape]
    cmaps = []
    keep_in_output = []

    dtype = coordmap.input_coords.coord_dtype
    newshape = []
    for i, __slice in enumerate(slices):
        ranges[i] = ranges[i][__slice]

        try:
            start = ranges[i][0]
        except IndexError:
            try:
                start = int(ranges[i])
            except TypeError:
                raise ValueError('empty slice for dimension %d, '
                                 'coordinate %s' % 
                                 (i, coordmap.input_coords.coord_names[i]))

        if ranges[i].shape == ():
            step = 0
            start = int(ranges[i])
            l = 1
        elif ranges[i].shape[0] > 1:
            start = ranges[i][0]
            step = ranges[i][1] - ranges[i][0]
            l = ranges[i].shape[0]
            keep_in_output.append(i)
        else:
            start = ranges[i][0]
            step = 0.
            l = 1
            keep_in_output.append(i)

        if step > 1:
            name = coordmap.input_coords.coord_names[i] + '-slice'
        else:
            name = coordmap.input_coords.coord_names[i]
        cmaps.append(Affine(
                np.array([[step, start],[0,1]], dtype=dtype), 
                CoordinateSystem([name], coord_dtype=dtype),
                CoordinateSystem([coordmap.input_coords.coord_names[i]])))
        if i in keep_in_output:
            newshape.append(l)
    slice_cmap = cmap_product(*cmaps)

    # Reduce the size of the matrix
    innames = slice_cmap.input_coords.coord_names
    inmat = []
    input_coords = CoordinateSystem(
        [innames[i] for i in keep_in_output],
        'input-slice',
        coordmap.input_coords.coord_dtype)
    A = np.zeros((coordmap.ndim[0]+1, len(keep_in_output)+1))
    for j, i in enumerate(keep_in_output):
        A[:,j] = slice_cmap.affine[:,i]
    A[:,-1] = slice_cmap.affine[:,-1]
    A = A.astype(input_coords.coord_dtype)
    slice_cmap = Affine(A, input_coords, coordmap.input_coords)
    return ArrayCoordMap(compose(coordmap, slice_cmap), tuple(newshape))
                   
class Grid(object):
    """
    Simple class to construct Affine instances with slice notation
    like np.ogrid/np.mgrid.

    >>> c = CoordinateSystem('xy', 'input')
    >>> g = Grid(c)
    >>> points = g[-1:1:21j,-2:4:31j]
    >>> points.coordmap.affine
    array([[ 0.1,  0. , -1. ],
           [ 0. ,  0.2, -2. ],
           [ 0. ,  0. ,  1. ]])

    >>> print points.coordmap.input_coords
    name: 'product', coord_names: ('i0', 'i1'), coord_dtype: float64
    >>> print points.coordmap.output_coords
    name: 'input', coord_names: ('x', 'y'), coord_dtype: float64

    >>> points.shape
    (21, 31)
    >>> print points.transposed_values.shape
    (2, 21, 31)
    >>> print points.values.shape
    (651, 2)
    """

    def __init__(self, coords):
        """
        Initialize Grid object

        Parameters
        ----------
        coords: ``CoordinateMap`` or ``CoordinateSystem``
           A coordinate map to be 'sliced' into. If
           coords is a CoordinateSystem, then an
           Affine instance is created with coords
           with identity transformation.
        """

        if isinstance(coords, CoordinateSystem):
            coordmap = Affine(np.identity(len(coords.coord_names)+1), coords, coords)
        elif not isinstance(coords, CoordinateMap):
            raise ValueError('expecting either a CoordinateMap or a CoordinateSystem for Grid')
        else:
            coordmap = coords
        self.coordmap = coordmap

    def __getitem__(self, index):
        """
        Create an Affine coordinate map with into self.coords with
        slices created as in np.mgrid/np.ogrid.
        """
        dtype = self.coordmap.input_coords.coord_dtype
        results = [a.ravel().astype(dtype) for a in np.ogrid[index]]
        if len(results) != len(self.coordmap.input_coords.coord_names):
            raise ValueError('the number of slice objects must match ' 
                             'the number of input dimensions')
        cmaps = []
        for i, result in enumerate(results):
            if result.shape[0] > 1:
                step = result[1] - result[0]
            else:
                step = 0
            start = result[0]
            cmaps.append(Affine(
                    np.array([[step, start],[0,1]], dtype=dtype), 
                    CoordinateSystem(['i%d' % i], coord_dtype=dtype),
                    CoordinateSystem([self.coordmap.input_coords.coord_names[i]],
                                     coord_dtype=dtype)))
        shape = [result.shape[0] for result in results]
        cmap = cmap_product(*cmaps)
        return ArrayCoordMap(compose(self.coordmap, cmap), tuple(shape))
