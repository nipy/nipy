"""
Coordinate maps store all the details about how an image translates to space.
They also provide mechanisms for iterating over that space.
"""
import copy, string

import numpy as np

from neuroimaging.core.reference.axis import Axis
from neuroimaging.core.reference.coordinate_system import CoordinateSystem, safe_dtype

__docformat__ = 'restructuredtext'

class CoordinateMap(object):
    """
    Defines a set of input and output coordinate systems and a mapping
    between the two, which represents the mapping of (for example) an
    image from voxel space to real space.
    """
    
    def __init__(self, mapping, input_coords, output_coords, inverse=None):
        """
        :Parameters:
            mapping : `callable`
                The mapping between input and output coordinates
            input_coords : `CoordinateSystem`
                The input coordinate system
            output_coords : `CoordinateSystem`
                The output coordinate system
            inverse : `callable`
                The optional 'inverse' of mapping, with the intention being
                x = inverse(mapping(x)). If the mapping is affine and invertible,
                then this is true for all x.
              
        """
        # These guys define the structure of the coordmap.
        self._mapping = mapping

        self.input_coords = input_coords
        self.output_coords = output_coords
        self._inverse_mapping = inverse

        if not callable(mapping):
            raise ValueError('mapping should be callable')

        if inverse is not None:
            if not callable(inverse):
                raise ValueError('if not None, inverse should be callable')
        self._checkmapping()

#     @staticmethod
#     def from_callable(innames, outnames, f, shape=None, inverse=None):
#         """
#         Construct a Mapping from a callable. If the
#         callable is an instance of Mapping, return it.
#         """
#         if not isinstance(f, Mapping):
#             f = Mapping(f, inverse=inverse)
#         if shape is None:
#             shape = [None]*len(innames)
#         return CoordinateMap(f, [Axis(n, length=l) for n, l in zip(innames, shape)],
#                              [Axis(n) for n in outnames])

    def _getmapping(self):
        return self._mapping
    mapping = property(_getmapping)

    def _getinverse_mapping(self):
        return self._inverse_mapping
    inverse_mapping = property(_getinverse_mapping)

    def _getinverse(self):
        """
        Return the inverse coordinate map.
        """
        if self._inverse_mapping is not None:
            return CoordinateMap(self._inverse_mapping, self.output_coords, self.input_coords, inverse=self.mapping)
    inverse = property(_getinverse)

    def _getshape(self):
        return tuple([len(a) for a in self.input_coords.axes])
    shape = property(_getshape)

    def _getndim(self):
        return (len(self.input_coords.axes), len(self.output_coords.axes))
    ndim = property(_getndim)

    def _checkshape(self, x):
        """
        Verify that x has the proper shape for evaluating the mapping
        """
        ndim = self.ndim

        if x.dtype.isbuiltin:
            if x.ndim > 2 or x.shape[-1] != ndim[0]:
                raise ValueError('if dtype is builtin, expecting a 2-d array of shape (*,%d) or a 1-d array of shape (%d,)' % (ndim[0], ndim[0]))
        elif x.ndim > 1:
            raise ValueError, 'if dtype is not builtin, expecting 1-d array, or a 0-d array' 

    def _checkmapping(self):
        """
        Verify that the input and output dimensions of self.mapping work.
        """
        input = np.zeros((10, self.ndim[0]), dtype=self.input_coords.builtin)
        output = self.mapping(input).astype(self.output_coords.builtin)
        if output.shape != (10, self.ndim[1]):
            raise ValueError('input and output dimensions of mapping do not agree with specified CoordinateSystems')

    def __call__(self, x):
        """
        Return mapping evaluated at x
        
        >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
        >>> inc = CoordinateSystem('input', inaxes)
        >>> outaxes = [Axis(x) for x in 'xyz']
        >>> outc = CoordinateSystem('output', outaxes)
        >>> cm = Affine(np.diag([1,2,3,1])), inc, outc)
        >>> cm([2,3,4])
        array([(2.0, 6.0, 12.0)])
        >>> cmi = cm.inverse
        >>> cmi([2,6,12])
        array([(2.0, 3.0, 4.0)])
        >>>                                    
        """
        return self.mapping(x)

    def copy(self):
        """
        Create a copy of the coordmap.

        :Returns: `CoordinateMap`
        """
        return CoordinateMap(self.mapping, self.input_coords,
                             self.output_coords, inverse=self.inverse_mapping)

    def __getitem__(self, index):
        """
        If hasattr(self, 'shape'), implying that input_coordinates
        are 'voxel coordinates', return a slice through the coordmap.

        Parameters
        ----------
        index : ``int`` or ``slice``
            sequence of integers or slices
        
        """

        if hasattr(self, 'shape'):
            return _slice(self, *index)
        else:
            raise AttributeError('CoordinateMap should have a shape in order to be sliced')

class Affine(CoordinateMap):
    """
    A class representing an affine transformation from an input coordinate system
    to an output coordinate system.
    
    This class has an affine property, which is a matrix representing
    the affine transformation in homogeneous coordinates. 
    This matrix is used to perform mappings,
    rather than having an explicit mapping function. 

    """

    def __init__(self, affine, input_coords, output_coords, dtype=None):
        """
        Return an CoordinateMap specified by an affine transformation in
        homogeneous coordinates.
        

        :Notes:

        The dtype of the resulting matrix is determined
        by finding a safe typecast for the input_coords, output_coords
        and affine.

        """

        dtype = safe_dtype(affine.dtype, input_coords.builtin, output_coords.builtin)

        inaxes = []
        for n in input_coords.axisnames:
            a = copy.copy(input_coords[n])
            a.dtype = dtype
            inaxes.append(a)

        outaxes = []
        for n in output_coords.axisnames:
            a = copy.copy(output_coords[n])
            a.dtype = dtype
            outaxes.append(a)

        self.input_coords = CoordinateSystem(input_coords.name, inaxes)
        self.output_coords = CoordinateSystem(output_coords.name, outaxes)
        self.affine = affine.astype(dtype)

        if self.affine.shape != (self.ndim[1]+1, self.ndim[0]+1):
            raise ValueError('coordinate lengths do not match affine matrix shape')

    def _getinverse_mapping(self):
        A, b = self.inverse.params
        def _mapping(x):
            value = np.dot(x, A.T)
            value += b
            return value
        return _mapping
    inverse_mapping = property(_getinverse_mapping)

    def _getmapping(self):
        A, b = self.params
        def _mapping(x):
            value = np.dot(x, A.T)
            value += b
            return value
        return _mapping
    mapping = property(_getmapping)

    def _getinverse(self):
        """
        Return the inverse coordinate map.
        """
        try:
            return Affine(np.linalg.inv(self.affine), self.output_coords, self.input_coords)
        except np.linalg.linalg.LinAlgError:
            pass
    inverse = property(_getinverse)

    def _getparams(self):
        return matvec_from_transform(self.affine)
    params = property(_getparams, doc='Get (matrix, vector) representation of affine.')

    def __call__(self, x):
        A, b = self.params
        value = np.dot(x, A.T)
        value += b
        return value

    @staticmethod
    def from_params(innames, outnames, params, shape=None):
        """
        Create an `Affine` instance from sequences of innames and outnames,
        as an optional shape.

        :Parameters:
            innames : ``tuple`` of ``string``
                The names of the axes of the input coordinate systems

            outnames : ``tuple`` of ``string``
                The names of the axes of the output coordinate systems

            params : `Affine`, `ndarray` or `(ndarray, ndarray)`
                An affine mapping between the input and output coordinate systems.
                This can be represented either by a single
                ndarray (which is interpreted as the representation of the
                mapping in homogeneous coordinates) or an (A,b) tuple.
            shape : ''tuple'' of ''int''
                The shape of the coordmap

        :Returns: `Affine`
        
        :Precondition: ``len(shape) == len(names)``
        
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        if type(params) == type(()):
            A, b = params
            params = transform_from_matvec(A, b)

        ndim = (len(innames) + 1, len(outnames) + 1)
        if params.shape != ndim:
            raise ValueError('shape and number of axis names do not agree')
        dtype = params.dtype

        if shape:
            inaxes = [Axis(name, length=l, dtype=dtype) for name, l in zip(innames, shape)]
        else:
            inaxes = [Axis(name, dtype=dtype) for name in innames]
        outaxes = [Axis(name, dtype=dtype) for name in outnames]
        input_coords = CoordinateSystem("input", inaxes)
        output_coords = CoordinateSystem('output', outaxes)
        return Affine(params, input_coords, output_coords)

    @staticmethod
    def from_start_step(innames, outnames, start, step, shape=None):
        """
        Create an `Affine` instance from sequences of names, shape, start
        and step.

        :Parameters:
            innames : ``tuple`` of ``string``
                The names of the axes of the input coordinate systems

            outnames : ``tuple`` of ``string``
                The names of the axes of the output coordinate systems

            start : ``tuple`` of ``float``
                Start vector used in constructing affine transformation
            step : ``tuple`` of ``float``
                Step vector used in constructing affine transformation
            shape : ''tuple'' of ''int''
                The shape of the coordmap

        :Returns: `CoordinateMap`
        
        :Predcondition: ``len(names) == len(start) == len(step)``
        """
        ndim = len(innames)
        if len(outnames) != ndim:
            raise ValueError, 'len(innames) != len(outnames)'

        cmaps = []
        for i in range(ndim):
            A = np.array([[step[i], start[i]],
                          [0, 1]])
            if shape:
                cmaps.append(Affine.from_params([innames[i]], [outnames[i]], A, (shape[i],)))
            else:
                cmaps.append(Affine.from_params([innames[i]], [outnames[i]], A))
        return product(*cmaps)

    @staticmethod
    def identity(names, shape=None):
        """
        Return an identity coordmap of the given shape.
        
        :Parameters:
            names : ``tuple`` of ``string``
                TODO
            dim :  ``int``
                TODO
            shape : ``tuple`` of ``int``
                TODO

        :Returns: `CoordinateMap` with `CoordinateSystem` input
                  and an identity transform 
        
        :Precondition: ``len(shape) == len(names)``
        
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        return Affine.from_start_step(names, names, [0]*len(names),
                                      [1]*len(names), shape=shape)

def _slice(coordmap, *slices):
    """
    Slice a 'voxel' CoordinateMap's input_coords with slices. A 'voxel' CoordinateMap
    is interpreted as a coordmap having a shape.

    """
    
    if len(slices) < coordmap.ndim[0]:
        slices = list(slices) + [slice(None,None,None)] * (coordmap.ndim[0] - len(slices))

    if not hasattr(coordmap, 'shape'):
        raise ValueError('CoordinateMap needs a shape to be sliced')

    ranges = [np.arange(len(a)) for a in coordmap.input_coords.axes]
    cmaps = []
    keep_in_output = []
    delete_from_output = []

    for i, __slice in enumerate(slices):
        print __slice
        ranges[i] = ranges[i][__slice]

        try:
            start = ranges[i][0]
        except IndexError:
            raise ValueError('empty slice for dimension %d, coordinate %s' % (i, coordmap.input_coords.axisnames[i]))

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

        cmaps.append(Affine.from_start_step([coordmap.input_coords.axisnames[i] + '-slice'],
                                            [coordmap.input_coords.axisnames[i]],
                                            [start],
                                            [step],
                                            shape=(l,)))
    slice_cmap = product(*cmaps)

    # Reduce the size of the matrix

    innames = slice_cmap.input_coords.axisnames
    inaxes = slice_cmap.input_coords.axes
    inmat = []
    input_coords = CoordinateSystem('input-slice', [Axis(innames[i], length=len(inaxes[i])) for i in keep_in_output])
    A = np.zeros((coordmap.ndim[0]+1, len(keep_in_output)+1))
    for j, i in enumerate(keep_in_output):
        A[:,j] = slice_cmap.affine[:,i]
    A[:,-1] = slice_cmap.affine[:,-1]
    slice_cmap = Affine(A, input_coords, coordmap.input_coords)
    return compose(coordmap, slice_cmap)
                   
def values(coordmap, transpose=False):
    """
    If the coordmap has a shape (so that it can be thought of as 
    a map from voxels to some output space), return
    the range of the coordmap, i.e. the value at all the voxels.

    :Inputs: 
        coordmap: `CoordinateMap`
        
        transposed: `bool`
            If False, the result is a 2-dimensional ndarray
            with shape[1] == coordmap.ndim[1]. That is,
            the result is a list of output values.

            Otherwise, the shape is (coordmap.ndim[1],) + coordmap.shape.

    :Returns: `ndarray`

    """

    if not hasattr(coordmap, 'shape'):
        raise ValueError('CoordinateMap needs a shape to compute the range')

    indices = np.indices(coordmap.shape)
    tmp_shape = indices.shape

    # reshape indices to be a sequence of coordinates
    indices.shape = (coordmap.ndim[0], np.product(coordmap.shape))
    _range = coordmap(indices.T)
    if transpose:
        _range = _range.T
        _range.shape = tmp_shape
    return _range 

def rename_input(coordmap, **kwargs):
    """
    Rename the input_coords, returning a new CoordinateMap

    >>> import numpy as np
    >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = CoordinateMap(Affine(np.identity(4)), inc, outc)
    >>> print cm.input_coords.values()
    [<Axis:"i", dtype=[('i', '<f8')], length=10>, <Axis:"j", dtype=[('j', '<f8')], length=20>, <Axis:"k", dtype=[('k', '<f8')], length=30>]
    >>> cm2 = rename_input(cm, i='x')
    >>> print cm2.input_coords
    {'axes': [<Axis:"x", dtype=[('x', '<f8')], length=10>, <Axis:"j", dtype=[('j', '<f8')], length=20>, <Axis:"k", dtype=[('k', '<f8')], length=30>], 'name': 'input-renamed'}
        
    """
    input_coords = coordmap.input_coords.rename(**kwargs)
    return CoordinateMap(coordmap.mapping, input_coords, coordmap.output_coords)

def rename_output(coordmap, **kwargs):
    """
    Rename the output_coords, returning a new CoordinateMap.
    
    >>> import numpy as np
    >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = CoordinateMap(Affine(np.identity(4)), inc, outc)
    >>> print cm.output_coords.values()
    [<Axis:"x", dtype=[('x', '<f8')]>, <Axis:"y", dtype=[('y', '<f8')]>, <Axis:"z", dtype=[('z', '<f8')]>]
    >>> cm2 = cm.rename_output(y='a')
    >>> print cm2.output_coords
    {'axes': [<Axis:"x", dtype=[('x', '<f8')]>, <Axis:"a", dtype=[('a', '<f8')]>, <Axis:"z", dtype=[('z', '<f8')]>], 'name': 'output-renamed'}

    >>>                             
    """
    output_coords = coordmap.output_coords.rename(**kwargs)
    return CoordinateMap(coordmap.mapping, coordmap.input_coords, output_coords)
        
def reorder_input(coordmap, order=None):
    """
    Create a new coordmap with reversed input_coords.
    Default behaviour is to reverse the order of the input_coords.
    If the coordmap has a shape, the resulting one will as well.

    Inputs:
    -------
    order: sequence
         Order to use, defaults to reverse. The elements
         can be integers, strings or 2-tuples of strings.
         If they are strings, they should be in coordmap.input_coords.axisnames.

    Returns:
    --------

    newcoordmap: `CoordinateMap`
         A new CoordinateMap with reversed input_coords.

    >>> inc = CoordinateSystem('input', inaxes)
    >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = CoordinateMap(Affine(np.identity(4)), inc, outc)
    >>> reorder_input(cm, 'ikj').shape
    (10, 30, 20)

    """
    ndim = coordmap.ndim[0]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [coordmap.input_coords.axisnames.index(s) for s in order]

    newaxes = [coordmap.input_coords.axes[i] for i in order]
    newincoords = CoordinateSystem(coordmap.input_coords.name + '-reordered', newaxes)
    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    A = CoordinateMap(Affine(perm), newincoords, coordmap.input_coords)
    return compose(coordmap, A)

def reorder_output(coordmap, order=None):
    """
    Create a new coordmap with reversed output_coords.
    Default behaviour is to reverse the order of the input_coords.
    
    Inputs:
    -------

    order: sequence
         Order to use, defaults to reverse. The elements
         can be integers, strings or 2-tuples of strings.
         If they are strings, they should be in coordmap.output_coords.axisnames.

    Returns:
    --------
        
    newcoordmap: `CoordinateMap`
         A new CoordinateMap with reversed output_coords.

    >>> inc = CoordinateSystem('input', inaxes)
    >>> inaxes = [Axis(x, length=l) for x, l in zip('ijk', (10,20,30))]
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = CoordinateMap(Affine(np.identity(4)), inc, outc)
    >>> reorder_output(cm, 'xzy').shape
    (10, 20, 30)
    >>> reorder_output(cm, [0,2,1]).shape
    (10, 20, 30)
    >>>                             

    >>> newcm = reorder_output(cm, 'yzx')
    >>> newcm.output_coords.axisnames
    ['y', 'z', 'x']
    >>>                              

    """

    ndim = coordmap.ndim[1]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [coordmap.output_coords.axisnames.index(s) for s in order]

    newaxes = [coordmap.output_coords.axes[i] for i in order]
    newoutcoords = CoordinateSystem(coordmap.output_coords.name + '-reordered', newaxes)
    
    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    A = CoordinateMap(Affine(perm), coordmap.output_coords, newoutcoords)
    return compose(A, coordmap)

def product(*cmaps):
    """
    Return the "topological" product of two or more CoordinateMaps.

    Inputs:
    -------
    cmaps : sequence of CoordinateMaps

    Returns:
    --------
    cmap : ``CoordinateMap``

    >>> inc1 = Affine.from_affine('i', 'x', np.diag([2,1]), (10,))
    >>> inc2 = Affine.from_affine('j', 'y', np.diag([3,1]), (20,))
    >>> inc3 = Affine.from_affine('k', 'z', np.diag([4,1]), (30,))

    >>> cmap = product(inc1, inc3, inc2)
    >>> cmap.shape
    (10, 30, 20)
    >>> cmap.input_coords.axisnames
    ['i', 'k', 'j']
    >>> cmap.output_coords.axisnames
    ['x', 'z', 'y']
    >>> cmap.affine
    array([[ 2.,  0.,  0.,  0.],
           [ 0.,  4.,  0.,  0.],
           [ 0.,  0.,  3.,  0.],
           [ 0.,  0.,  0.,  1.]])

    """
    inaxes = []
    outaxes = []
    innames = []
    outnames = []
    ndimin = []

    for cmap in cmaps:
        inaxes += cmap.input_coords.axes
        outaxes += cmap.output_coords.axes
        innames += [cmap.input_coords.name]
        outnames += [cmap.output_coords.name]
        ndimin.append(cmap.ndim[0])

    ndimin.insert(0,0)
    ndimin = tuple(np.cumsum(ndimin))
    innames = string.join(innames, ' * ')
    outnames = string.join(outnames, ' * ')

    def mapping(x):
        x = np.asarray(x)
        y = []
        for i in range(len(ndimin)-1):
            cmap = cmaps[i]
            if x.ndim == 2:
                yy = cmaps[i](x[:,ndimin[i]:ndimin[i+1]])
            else:
                yy = cmaps[i](x[ndimin[i]:ndimin[i+1]])
            y.append(yy)
        yy = np.hstack(y)
        return yy

    notaffine = filter(lambda x: not isinstance(x, Affine), cmaps)
    if not notaffine:
        affine = linearize(mapping, ndimin[-1])
        return Affine(affine, CoordinateSystem(innames, inaxes),
                      CoordinateSystem(outnames, outaxes))
    return CoordinateMap(mapping, CoordinateSystem(innames, inaxes),
                      CoordinateSystem(outnames, outaxes))

def compose(*cmaps):
    """
    Return the (right) composition of two or more CoordinateMaps.

    Inputs:
    -------
    cmaps : sequence of CoordinateMaps

    Returns:
    --------
    cmap : ``CoordinateMap``
         The resulting CoordinateMap has input_coords == cmaps[-1].input_coords
         and output_coords == cmaps[0].output_coords

    >>> cmap = CoordinateMap.from_affine('i', 'x', Affine(np.diag([2,1])), (10,))
    >>> cmapi = cmap.inverse
    >>> id1 = compose(cmap,cmapi)
    >>> print id1.affine
    [[ 1.  0.]
     [ 0.  1.]]
    >>> assert not hasattr(id1, 'shape')
    >>> id2 = compose(cmapi,cmap)
    >>> assert id2.shape == (10,)
    >>> id1.input_coords.axisnames
    ['x']
    >>> id2.input_coords.axisnames
    ['i']
    >>> 

    """

    def _compose2(cmap1, cmap2):
        forward = lambda input: cmap1.mapping(cmap2.mapping(input))
        if cmap1.inverse is not None and cmap2.inverse is not None:
            backward = lambda output: cmap2.inverse.mapping(cmap1.inverse.mapping(output))
        else:
            backward = None
        return forward, backward

    cmap = cmaps[-1]
    for i in range(len(cmaps)-2,-1,-1):
        m = cmaps[i]
        if m.input_coords == cmap.output_coords:
            forward, backward = _compose2(m, cmap)
            cmap = CoordinateMap(forward, cmap.input_coords, m.output_coords, inverse=backward)
        else:
            raise ValueError, 'input and output coordinates do not match: input=%s, output=%s' % (`m.input_coords.dtype`, `cmap.output_coords.dtype`)

    notaffine = filter(lambda cmap: not isinstance(cmap, Affine), cmaps)
    print map(lambda cmap: not isinstance(cmap, Affine), cmaps)
    if not notaffine:
        affine = linearize(cmap, cmap.ndim[0])
        return Affine(affine, cmap.input_coords,
                      cmap.output_coords)
    return cmap
    
def replicate(coordmap, n, concataxis='string'):
    """
    Create a CoordinateMap by taking the product
    of coordmap with a 1-dimensional 'concat' CoordinateSystem

    :Parameters:
         coordmap : `CoordinateMap`
                The coordmap to be used
         n : ``int``
                The number of tiems to concatenate the coordmap
         concataxis : ``string``
                The name of the new dimension formed by concatenation
    """
    concat = CoordinateMap.from_affine([concataxis], [concataxis], Affine(np.identity(2)), (n,))
    return product(concat, coordmap)

#TODO: renames this interpolate? And implement interpolation?
def hstack(*cmaps):
    """
    Return a "hstacked" CoordinateMap. That is,
    take the result of a number of cmaps, and return np.hstack(results)
    with an additional first row being the 'concat' axis values.

    If the cmaps are identical
    the resulting map is essentially
    replicate(cmaps[0], len(cmaps)) but the mapping is not Affine.

    Some simple modifications of this function would allow 'interpolation'
    along the 'concataxis'. 

    Inputs:
    -------
    cmaps : sequence of CoordinateMaps
          Each cmap should have the same input_coords, output_coords and shape.

    Returns:
    --------
    cmap : ``CoordinateMap``

    >>> inc1 = CoordinateMap.from_affine('ab', 'cd', Affine(np.diag([2,3,1])), (10,20))
    >>> inc2 = CoordinateMap.from_affine('ab', 'cd', Affine(np.diag([3,2,1])), (10,20))
    >>> inc3 = CoordinateMap.from_affine('ab', 'cd', Affine(np.diag([1,1,1])), (10,20))
    >>> stacked = hstack(inc1, inc2, inc3)

    >>> stacked(np.array([[0,1,2],[1,1,2],[2,1,2], [1,1,2]]).T)
    array([[ 0.,  2.,  6.],
           [ 1.,  3.,  4.],
           [ 2. , 1.,  2.],
           [ 1.,  3.,  4.]])
    >>> 

    """

    # Ensure that they all have the same coordinate systems

    notinput = filter(lambda i: cmaps[i].input_coords != cmaps[0].input_coords, range(len(cmaps)))
    notoutput = filter(lambda i: cmaps[i].output_coords != cmaps[0].output_coords, range(len(cmaps)))
    notshape = filter(lambda i: cmaps[i].shape != cmaps[0].shape, range(len(cmaps)))

    if notinput or notoutput or notshape:
        raise ValueError("input and output coordinates as well as shape of each CoordinateMap should be the same in order to stack them")

    def mapping(x, return_index=False):
        r = []
        for i in range(x.shape[1]):
            ii = int(x[0,i])
            y = cmaps[ii](x[1:,i])
            r.append(np.hstack([x[0,i], y]))
        return np.vstack(r)

    stackin = Axis('stack-input', length=len(cmaps))
    stackout = Axis('stack-output')

    inaxes = [stackin] + cmaps[0].input_coords.axes
    incoords = CoordinateSystem('stackin-%s' % cmaps[0].input_coords.name, 
                                inaxes)
    outaxes = [stackout] + cmaps[0].output_coords.axes
    outcoords = CoordinateSystem('stackout-%s' % cmaps[0].output_coords.name, 
                                 outaxes)
    return CoordinateMap(mapping, incoords, outcoords)

def matvec_from_transform(transform):
    """ Split a tranformation represented in homogeneous
    coordinates into it's matrix and vector components. """
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return matrix, vector

def transform_from_matvec(matrix, vector):
    """ Combine a matrix and vector into its representation in homogeneous coordinates. """
    nin, nout = matrix.shape
    t = np.zeros((nin+1,nout+1), matrix.dtype)
    t[0:nin, 0:nout] = matrix
    t[nin,   nout] = 1.
    t[0:nin, nout] = vector
    return t


def linearize(mapping, ndimin, step=np.array(1.), origin=None):
    """
    Given a Mapping of ndimin variables, 
    with an input builtin dtype, return the linearization
    of mapping at origin based on a given step size
    in each coordinate axis.

    If not specified, origin defaults to np.zeros(ndimin, dtype=dtype).
    
    :Inputs: 
        mapping: ``Mapping``
              A function to linearize
        ndimin: ``int``
              Number of input dimensions to mapping
        origin: ``ndarray``
              Origin at which to linearize mapping
        step: ``ndarray``
              Step size, an ndarray with step.shape == ().

    :Returns:
        C: ``ndarray``
            Linearization of mapping in homogeneous coordinates, i.e. 
            an array of size (ndimout+1, ndimin+1) where
            ndimout = mapping(origin).shape[0].

    :Notes: The dtype of the resulting Affine mapping
            will be the dtype of mapping(origin)/step, regardless
            of the input dtype.

    """
    step = np.asarray(step)
    if step.shape != ():
        raise ValueError('step should be a scalar value')
    if origin is None:
        origin = np.zeros(ndimin, step.dtype)
    else:
        if origin.dtype != step.dtype:
            warnings.warn('origin.dtype != step.dtype in function linearize, using step.dtype')
        origin = np.asarray(origin, dtype=step.dtype)
        if origin.shape != (ndimin,):
            raise ValueError('origin.shape != (%d,)' % ndimin)
    b = mapping(origin)

    origin = np.multiply.outer(np.ones(ndimin), origin)
    y1 = mapping(step*np.identity(ndimin) + origin)
    y0 = mapping(origin)

    ndimout = y1.shape[1]
    C = np.zeros((ndimout+1, ndimin+1), (y0/step).dtype)
    C[-1,-1] = 1
    C[:ndimout,-1] = b
    C[:ndimout,:ndimin] = (y1 - y0).T / step
    return C

class Grid():
    """
    Simple class to construct Affine instances with slice notation like np.ogrid/np.mgrid

    >>> c = CoordinateSystem('input', [Axis(n) for n in 'xy'])
    >>> g = Grid(c)
    >>> points = g[-1:1:21j,-2:4:31j]
    >>> points.affine
    array([[ 0.1,  0. , -1. ],
           [ 0. ,  0.2, -2. ],
           [ 0. ,  0. ,  1. ]])

    >>> print points.input_coords
    {'axes': [<Axis:"i0", dtype=[('i0', '<f8')]>, <Axis:"i1", dtype=[('i1', '<f8')]>], 'name': 'i0 * i1'}
    >>> print points.output_coords
    {'axes': [<Axis:"x", dtype=[('x', '<f8')]>, <Axis:"y", dtype=[('y', '<f8')]>], 'name': 'x * y'}
    >>>                                                  

    >>> points.shape
    (21, 31)
    >>> print values(points, transpose=True).shape
    (2, 21, 31)
    >>> print values(points, transpose=False).shape
    (651, 2)
    """

    def __init__(self, coords):

        """
        :Inputs: 
           coords: ``CoordinateSystem``
               A coordinate system to be 'sliced' into
        """

        self.coords = coords

    def __getitem__(self, index):
        """
        Create an Affine coordinate map with into self.coords with
        slices created as in np.mgrid/np.ogrid.
        """
        results = [a.ravel() for a in np.ogrid[index]]
        if len(results) != len(self.coords.axisnames):
            raise ValueError('must slice all axes to create a grid')

        cmaps = []
        for i, result in enumerate(results):
            if result.shape[0] > 1:
                step = result[1] - result[0]
            else:
                step = 0
            start = result[0]
            cmaps.append(Affine(np.array([[step, start],[0,1]]), 
                                CoordinateSystem('i%d' % i, [Axis('i%d' % i, length=result.shape[0])]),
                                CoordinateSystem(self.coords.axisnames[i], [self.coords.axes[i]])))
        return product(*cmaps)
