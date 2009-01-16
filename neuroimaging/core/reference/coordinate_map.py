"""
Coordinate maps store all the details about how an image translates to space.
They also provide mechanisms for iterating over that space.
"""
import copy, string

import numpy as np
from neuroimaging.core.reference.coordinate_system import CoordinateSystem, Coordinate, safe_dtype 

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

    def _checkmapping(self, check_outdtype=True):
        """
        Verify that the input and output dimensions of self.mapping work.

        Also
        """
        input = np.zeros((10, self.ndim[0]), dtype=self.input_coords.builtin)
        output = self.mapping(input)
        if output.dtype != self.output_coords.builtin and check_outdtype:
            warnings.warn('output.dtype != self.output_coords.builtin')
        output = output.astype(self.output_coords.builtin)
        if output.shape != (10, self.ndim[1]):
            raise ValueError('input and output dimensions of mapping do not agree with specified CoordinateSystems')

    def __call__(self, x):
        """
        Return mapping evaluated at x
        
        >>> inaxes = [Axis(x) for x in 'ijk']
        >>> inc = CoordinateSystem('input', inaxes)
        >>> outaxes = [Axis(x) for x in 'xyz']
        >>> outc = CoordinateSystem('output', outaxes)
        >>> cm = Affine(np.diag([1,2,3,1]), inc, outc)
        >>> cm([2,3,4])
        array([  2.,   6.,  12.])
        >>> cmi = cm.inverse
        >>> cmi([2,6,12])
        array([ 2.,  3.,  4.])
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

    def copy(self):
        """
        Create a copy of the coordmap.

        :Returns: `CoordinateMap`
        """
        return Affine(self.affine, self.input_coords,
                      self.output_coords)


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
    def from_params(innames, outnames, params):
        """
        Create an `Affine` instance from sequences of innames and outnames.

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

        :Returns: `Affine`
        
        :Precondition: ``len(shape) == len(names)``
        
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        if type(params) == type(()):
            A, b = params
            params = transform_from_matvec(A, b)

        ndim = (len(innames) + 1, len(outnames) + 1)
        if params.shape != ndim[::-1]:
            raise ValueError('shape and number of axis names do not agree')
        dtype = params.dtype

        inaxes = [Coordinate(name, dtype=dtype) for name in innames]
        outaxes = [Coordinate(name, dtype=dtype) for name in outnames]
        input_coords = CoordinateSystem("input", inaxes)
        output_coords = CoordinateSystem('output', outaxes)
        return Affine(params, input_coords, output_coords)

    @staticmethod
    def from_start_step(innames, outnames, start, step):
        """
        Create an `Affine` instance from sequences of names, start
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
            cmaps.append(Affine.from_params([innames[i]], [outnames[i]], A))
        return product(*cmaps)

    @staticmethod
    def identity(names):
        """
        Return an identity coordmap of the given shape.
        
        :Parameters:
            names : ``tuple`` of ``string`` 
                  Names of Axes in output CoordinateSystem

        :Returns: `CoordinateMap` with `CoordinateSystem` input
                  and an identity transform, with identical input and output coords. 
        
        """
        return Affine.from_start_step(names, names, [0]*len(names),
                                      [1]*len(names))

def rename_input(coordmap, **kwargs):
    """
    Rename the input_coords, returning a new CoordinateMap

    >>> import numpy as np
    >>> inaxes = [Axis(x) for x in 'ijk']
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = Affine(np.identity(4), inc, outc)
    >>> print cm.input_coords.values()
    [<Axis:"i", dtype=[('i', '<f8')]>, <Axis:"j", dtype=[('j', '<f8')]>, <Axis:"k", dtype=[('k', '<f8')]>]
    >>> cm2 = rename_input(cm, i='x')
    >>> print cm2.input_coords
    {'axes': [<Axis:"x", dtype=[('x', '<f8')]>, <Axis:"j", dtype=[('j', '<f8')]>, <Axis:"k", dtype=[('k', '<f8')]>], 'name': 'input-renamed'}
        
    """
    input_coords = coordmap.input_coords.rename(**kwargs)
    return CoordinateMap(coordmap.mapping, input_coords, coordmap.output_coords)

def rename_output(coordmap, **kwargs):
    """
    Rename the output_coords, returning a new CoordinateMap.
    
    >>> import numpy as np
    >>> inaxes = [Axis(x) for x in 'ijk']
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = Affine(np.identity(4), inc, outc)
    >>> print cm.output_coords.values()
    [<Axis:"x", dtype=[('x', '<f8')]>, <Axis:"y", dtype=[('y', '<f8')]>, <Axis:"z", dtype=[('z', '<f8')]>]
    >>> cm2 = rename_output(cm, y='a')
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

    >>> inaxes = [Axis(x) for x in 'ijk']
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = Affine(np.identity(4), inc, outc)
    >>> print reorder_input(cm, 'ikj').input_coords
    {'axes': [<Axis:"i", dtype=[('i', '<f8')]>, <Axis:"k", dtype=[('k', '<f8')]>, <Axis:"j", dtype=[('j', '<f8')]>], 'name': 'input-reordered'}

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

    perm = perm.astype(coordmap.input_coords.builtin)
    A = Affine(perm, newincoords, coordmap.input_coords)
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

    >>> inaxes = [Axis(x) for x in 'ijk']
    >>> inc = CoordinateSystem('input', inaxes)
    >>> outaxes = [Axis(x) for x in 'xyz']
    >>> outc = CoordinateSystem('output', outaxes)
    >>> cm = Affine(np.identity(4), inc, outc)
    >>> print reorder_output(cm, 'xzy').output_coords
    {'axes': [<Axis:"x", dtype=[('x', '<f8')]>, <Axis:"z", dtype=[('z', '<f8')]>, <Axis:"y", dtype=[('y', '<f8')]>], 'name': 'output-reordered'}
    >>> print reorder_output(cm, [0,2,1]).output_coords.axisnames
    ['x', 'z', 'y']
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

    perm = perm.astype(coordmap.output_coords.builtin)
    A = Affine(perm, coordmap.output_coords, newoutcoords)
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

    >>> inc1 = Affine.from_params('i', 'x', np.diag([2,1]))
    >>> inc2 = Affine.from_params('j', 'y', np.diag([3,1]))
    >>> inc3 = Affine.from_params('k', 'z', np.diag([4,1]))

    >>> cmap = product(inc1, inc3, inc2)
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

    incoords, outcoords = CoordinateSystem(innames, inaxes), CoordinateSystem(outnames, outaxes)
    if not notaffine:

        affine = linearize(mapping, ndimin[-1], step=np.array(1, incoords.builtin))
        return Affine(affine, incoords, outcoords)
    return CoordinateMap(mapping, incoords, outcoords)

def compose(*cmaps):
    """
    Return the composition of two or more CoordinateMaps.

    Inputs:
    -------
    cmaps : sequence of CoordinateMaps

    Returns:
    --------
    cmap : ``CoordinateMap``
         The resulting CoordinateMap has input_coords == cmaps[-1].input_coords
         and output_coords == cmaps[0].output_coords

    >>> cmap = Affine.from_params('i', 'x', np.diag([2.,1.]))
    >>> cmapi = cmap.inverse
    >>> id1 = compose(cmap,cmapi)
    >>> print id1.affine
    [[ 1.  0.]
     [ 0.  1.]]

    >>> id2 = compose(cmapi,cmap)
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

    if not notaffine:
        affine = linearize(cmap, cmap.ndim[0], step=np.array(1, cmaps[0].output_coords.builtin))
        return Affine(affine, cmap.input_coords,
                      cmap.output_coords)
    return cmap
    
def replicate(coordmap, n, concataxis='concat'):
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
          Each cmap should have the same input_coords and output_coords.

    Returns:
    --------
    cmap : ``CoordinateMap``

    >>> inc1 = Affine.from_params('ab', 'cd', np.diag([2,3,1]))
    >>> inc2 = Affine.from_params('ab', 'cd', np.diag([3,2,1]))
    >>> inc3 = Affine.from_params('ab', 'cd', np.diag([1,1,1]))
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

    if notinput or notoutput:
        raise ValueError("input and output coordinates of each CoordinateMap should be the same in order to stack them")

    def mapping(x, return_index=False):
        r = []
        for i in range(x.shape[1]):
            ii = int(x[0,i])
            y = cmaps[ii](x[1:,i])
            r.append(np.hstack([x[0,i], y]))
        return np.vstack(r)

    stackin = Coordinate('stack-input')
    stackout = Coordinate('stack-output')

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
    dtype = step.dtype
    if step.shape != ():
        raise ValueError('step should be a scalar value')
    if origin is None:
        origin = np.zeros(ndimin, dtype)
    else:
        if origin.dtype != step.dtype:
            warnings.warn('origin.dtype != step.dtype in function linearize, using step.dtype')
        origin = np.asarray(origin, dtype=step.dtype)
        if origin.shape != (ndimin,):
            raise ValueError('origin.shape != (%d,)' % ndimin)
    b = mapping(origin)

    origin = np.multiply.outer(np.ones(ndimin, dtype), origin)
    y1 = mapping(step*np.identity(ndimin) + origin)
    y0 = mapping(origin)

    ndimout = y1.shape[1]
    C = np.zeros((ndimout+1, ndimin+1), (y0/step).dtype)
    C[-1,-1] = 1
    C[:ndimout,-1] = b
    C[:ndimout,:ndimin] = (y1 - y0).T / step
    return C

