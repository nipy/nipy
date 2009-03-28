"""
CoordinateMaps map (transform) an image from an input space to an output space.

A CoordinateMap object contains all the details about an input
CoordinateSystem, an output CoordinateSystem and the mappings between
them.  The *mapping* transforms an image from the input coordinate
system to the output coordinate system.  And the *inverse_mapping*
performs the opposite transformation.  The *inverse_mapping* can be
specified explicity when creating the CoordinateMap or implicitly in the
case of an Affine CoordinateMap.

"""

"""
Matthew, Cindee thoughts

Should we change the order of input args to CoordinateMap to:

CoordinateMap(input_coords, output_coords, mapping, inverse_mapping=None)
or keep as
CoordinateMap(mapping, input_coords, output_coords, inverse_mapping=None)

CoordinateMap.ndim should be renamed to CoordinateMap.ndims, because it returns more than one value.

Affine should be renamed to AffineMap, or AffineCoordMap, or something

We need to think about whether coordinate values should be N by 3
(where 3 is the number of coordinates in the coordinate system), or 3
by N.

3 by N makes more sense when we think of applying an affine matrix to the points.

We might think, that if we have a matrix ``arr`` with points in, then
``arr[0]`` should be the first point [x,y,z], rather then the first coordinate
value of all the points - N values of [x]

Consider renaming reorder_input to self.input_reordered() or
something.  reorder_output similarly.

That's as far as we got.

"""


import warnings

import numpy as np

import neuroimaging.core.transforms.affines as affines
from neuroimaging.core.reference.coordinate_system import(CoordinateSystem, 
                                                          safe_dtype)
from neuroimaging.core.reference.coordinate_system import product as coordsys_product

__docformat__ = 'restructuredtext'

class CoordinateMap(object):
    """A set of input and output CoordinateSystems and a mapping between them.

    For example, the mapping may represent the mapping of an image
    from voxel space (the input coordinates) to real space (the output
    coordinates).  The mapping may be an affine or non-affine
    transformation.

    Attributes
    ----------
    input_coords : :class:`CoordinateSystem`
        The input coordinate system.
    output_coords : :class:`CoordinateSystem`
        The output coordinate system.
    mapping : callable
        A callable that maps the input_coords to the output_coords.
    inverse_mapping : None or callable
        A callable that maps the output_coords to the input_coords.
        Not all mappings have an inverse, in which case
        inverse_mapping is None.
        
    Examples
    --------
    >>> input_coords = CoordinateSystem('ijk', 'voxels')
    >>> output_coords = CoordinateSystem('xyz', 'world')
    >>> mni_orig = np.array([-90.0, -126.0, -72.0])
    >>> mapping = lambda x: x + mni_orig
    >>> inv_mapping = lambda x: x - mni_orig
    >>> cm = CoordinateMap(mapping, input_coords, output_coords, inv_mapping)

    Map the first 3 voxel coordinates, along the x-axis, to mni space:

    >>> x = np.array([[0,0,0], [1,0,0], [2,0,0]])
    >>> cm.mapping(x)
    array([[ -90., -126.,  -72.],
           [ -89., -126.,  -72.],
           [ -88., -126.,  -72.]])


    """
    def __init__(self, mapping, 
                 input_coords, 
                 output_coords, 
                 inverse_mapping=None):
        """Create a CoordinateMap given the input/output coords and mappings.

        Parameters
        ----------
        mapping : callable
           The mapping between input and output coordinates
        input_coords : :class:`CoordinateSystem`
           The input coordinate system
        output_coords : :class:`CoordinateSystem`
           The output coordinate system
        inverse_mapping : None or callable, optional
           The optional inverse of mapping, with the intention being
           ``x = inverse_mapping(mapping(x))``.  If the mapping is
           affine and invertible, then this is true for all x.  The
           default is None

        Returns
        -------
        coordmap : CoordinateMap
        """
        # These attrs define the structure of the coordmap.
        self._mapping = mapping
        self._input_coords = input_coords
        self._output_coords = output_coords
        self._inverse_mapping = inverse_mapping

        if not callable(mapping):
            raise ValueError('The mapping must be callable.')
        if inverse_mapping is not None:
            if not callable(inverse_mapping):
                raise ValueError('The inverse_mapping must be callable.')
        self._checkmapping()

    @property
    def input_coords(self):
        'input coordinate system'
        return self._input_coords

    @property
    def output_coords(self):
        'output coordinate system'
        return self._output_coords

    @property
    def mapping(self):
        'The mapping from input_coords to output_coords.'
        return self._mapping

    @property
    def inverse_mapping(self):
        'The mapping from output_coords to input_coords'
        return self._inverse_mapping

    @property
    def inverse(self):
        """
        Return a new CoordinateMap with the mappings reversed
        """
        if self._inverse_mapping is None:
            return None
        return CoordinateMap(self._inverse_mapping, 
                             self._output_coords, 
                             self._input_coords, 
                             inverse_mapping=self._mapping)

    @property
    def ndim(self):
        'Number of dimensions of input and output coordinates.'
        return (self._input_coords.ndim, self._output_coords.ndim)

    def _checkmapping(self):
        """Verify that the input and output dimensions of self.mapping work.

        We do this by passing something that should work, through __call__
        """
        inp = np.zeros((10, self.ndim[0]),
                       dtype=self._input_coords.coord_dtype)
        out = self(inp)

    def __call__(self, x):
        """Return mapping evaluated at x

        Check input and output of mapping for compatiblity with input
        and output coordinate systems respectively.

        Parameters
        ----------
        x : array-like
           Values in input coordinate system space that will be mapped
           to the output coordinate system space, using
           ``self.mapping``
           
        Returns
        -------
        y : array
           Values in output coordinate system space

        Examples
        --------
        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> mapping = lambda x:x+1
        >>> inverse = lambda x:x-1
        >>> cm = CoordinateMap(mapping, input_cs, output_cs, inverse)
        >>> cm([2,3,4])
        array([[3, 4, 5]])
        >>> cmi = cm.inverse
        >>> cmi([2,6,12])
        array([[ 1,  5, 11]])

        """

        in_vals = self._input_coords._checked_values(x)
        out_vals = self._mapping(in_vals)
        return self._output_coords._checked_values(out_vals)

    def copy(self):
        """Create a copy of the coordmap.

        Returns
        -------
        coordmap : CoordinateMap

        """

        return CoordinateMap(self._mapping, 
                             self._input_coords,
                             self._output_coords, 
                             inverse_mapping=self._inverse_mapping)

class Affine(CoordinateMap):
    """
    A class representing an affine transformation from an input
    coordinate system to an output coordinate system.
    
    This class has an affine property, which is a matrix representing
    the affine transformation in homogeneous coordinates.  This matrix
    is used to perform mappings, rather than having an explicit
    mapping function.

    >>> inp_cs = CoordinateSystem('ijk')
    >>> out_cs = CoordinateSystem('xyz')
    >>> cm = Affine(np.diag([1, 2, 3, 1]), inp_cs, out_cs)
    >>> cm.affine
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  2.,  0.,  0.],
           [ 0.,  0.,  3.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> cm([1,1,1])
    array([[ 1.,  2.,  3.]])
    >>> icm = cm.inverse
    >>> icm([1,2,3])
    array([[ 1.,  1.,  1.]])
    """

    def __init__(self, affine, input_coords, output_coords):
        """
        Return an CoordinateMap specified by an affine transformation
        in homogeneous coordinates.
        
        Parameters
        ----------
        affine : array-like
           affine homogenous coordinate matrix
        input_coords : :class:`CoordinateSystem`
           input coordinates
        output_coords : :class:`CoordinateSystem`
           output coordinates

        Notes
        -----
        The dtype of the resulting matrix is determined by finding a
        safe typecast for the input_coords, output_coords and affine.
        """
        dtype = safe_dtype(affine.dtype,
                           input_coords.coord_dtype,
                           output_coords.coord_dtype)
        inaxes = input_coords.coord_names
        outaxes = output_coords.coord_names
        self._input_coords = CoordinateSystem(inaxes,
                                              input_coords.name,
                                              dtype)
        self._output_coords = CoordinateSystem(outaxes,
                                               output_coords.name,
                                               dtype)
        affine = np.asarray(affine, dtype=dtype)
        if affine.shape != (self.ndim[1]+1, self.ndim[0]+1):
            raise ValueError('coordinate lengths do not match '
                             'affine matrix shape')
        self._affine = affine
        A, b = affines.to_matrix_vector(affine)
        def _mapping(x):
            value = np.dot(x, A.T)
            value += b
            return value
        self._mapping = _mapping

    @property
    def affine(self):
        """The affine transform matrix of the Affine CoordinateMap."""
        return self._affine
    
    @property
    def inverse_mapping(self):
        """The inverse affine mapping from the Affine CoordinateMap."""
        inverse = self.inverse
        if inverse is None:
            raise ValueError('There is no inverse for this affine')
        return inverse.mapping

    @property
    def inverse(self):
        """
        Return the inverse coordinate map.
        """
        try:
            return Affine(np.linalg.inv(self.affine), 
                          self.output_coords, 
                          self.input_coords)
        except np.linalg.linalg.LinAlgError:
            pass

    @staticmethod
    def from_params(innames, outnames, params):
        """
        Create an `Affine` instance from sequences of innames and outnames.

        Parameters
        ----------
        innames : ``tuple`` of ``string``
           The names of the axes of the input coordinate systems
        outnames : ``tuple`` of ``string``
           The names of the axes of the output coordinate systems
        params : `Affine`, `ndarray` or `(ndarray, ndarray)`
           An affine mapping between the input and output coordinate
           systems.  This can be represented either by a single
           ndarray (which is interpreted as the representation of the
           mapping in homogeneous coordinates) or an (A,b) tuple.

        Returns
        -------
        aff : `Affine` object instance
        
        Notes
        -----
        :Precondition: ``len(shape) == len(names)``
        
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        if type(params) == type(()):
            A, b = params
            params = affines.from_matrix_vector(A, b)

        ndim = (len(innames) + 1, len(outnames) + 1)
        if params.shape != ndim[::-1]:
            raise ValueError('shape and number of axis names do not agree')
        dtype = params.dtype

        input_coords = CoordinateSystem(innames, "input")
        output_coords = CoordinateSystem(outnames, 'output')
        return Affine(params, input_coords, output_coords)

    @staticmethod
    def from_start_step(innames, outnames, start, step):
        """
        Create an `Affine` instance from sequences of names, start
        and step.

        Parameters
        ----------
        innames : ``tuple`` of ``string``
            The names of the axes of the input coordinate systems
        outnames : ``tuple`` of ``string``
            The names of the axes of the output coordinate systems
        start : ``tuple`` of ``float``
            Start vector used in constructing affine transformation
        step : ``tuple`` of ``float``
            Step vector used in constructing affine transformation

        Returns
        -------
        cm : `CoordinateMap`

        Examples
        --------
        >>> cm = Affine.from_start_step('ijk', 'xyz', [1, 2, 3], [4, 5, 6])
        >>> cm.affine
        array([[ 4.,  0.,  0.,  1.],
               [ 0.,  5.,  0.,  2.],
               [ 0.,  0.,  6.,  3.],
               [ 0.,  0.,  0.,  1.]])
        
        Notes
        -----
        ``len(names) == len(start) == len(step)``
        
        """
        ndim = len(innames)
        if len(outnames) != ndim:
            raise ValueError('len(innames) != len(outnames)')
        return Affine.from_params(innames, 
                                  outnames, 
                                  (np.diag(step), start))

    @staticmethod
    def identity(names):
        """
        Return an identity coordmap of the given shape.
        
        Parameters
        ----------
        names : ``tuple`` of ``string`` 
           Names of Axes in output CoordinateSystem

        Returns
        -------
        cm : `CoordinateMap` 
           ``CoordinateMap`` with `CoordinateSystem` input and an
           identity transform, with identical input and output coords.

        Examples
        --------
        >>> cm = Affine.identity('ijk')
        >>> cm.affine
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])
        >>> print cm.input_coords
        name: 'input', coord_names: ('i', 'j', 'k'), coord_dtype: float64
        >>> print cm.output_coords
        name: 'output', coord_names: ('i', 'j', 'k'), coord_dtype: float64
        """
        return Affine.from_start_step(names, names, [0]*len(names),
                                      [1]*len(names))

    def copy(self):
        """
        Create a copy of the coordmap.

        Returns
        -------
        cm : `CoordinateMap`

        Examples
        --------
        >>> cm = Affine(np.eye(4), CoordinateSystem('ijk'), CoordinateSystem('xyz'))
        >>> cm_copy = cm.copy()
        >>> cm is cm_copy
        False

        Note that the matrix (affine) is not a pointer to the
        same data, it's a full independent copy

        >>> cm.affine[0,0] = 2.0
        >>> cm_copy.affine[0,0]
        1.0
        """
        return Affine(self._affine.copy(), self._input_coords,
                      self._output_coords)


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
         If they are strings, they should be in coordmap.input_coords.coord_names.

    Returns:
    --------

    newcoordmap: `CoordinateMap`
         A new CoordinateMap with reversed input_coords.

    >>> input_cs = CoordinateSystem('ijk')
    >>> output_cs = CoordinateSystem('xyz')
    >>> cm = Affine(np.identity(4), input_cs, output_cs)
    >>> print reorder_input(cm, 'ikj').input_coords
    name: '-reordered', coord_names: ('i', 'k', 'j'), coord_dtype: float64
    """
    ndim = coordmap.ndim[0]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [coordmap.input_coords.index(s) for s in order]

    newaxes = [coordmap.input_coords.coord_names[i] for i in order]
    newincoords = CoordinateSystem(newaxes, 
                                   coordmap.input_coords.name + '-reordered', 
                                   coord_dtype=coordmap.input_coords.coord_dtype)
    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    perm = perm.astype(coordmap.input_coords.coord_dtype)
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
         If they are strings, they should be in coordmap.output_coords.coord_names.

    Returns:
    --------
        
    newcoordmap: `CoordinateMap`
         A new CoordinateMap with reversed output_coords.

    >>> input_cs = CoordinateSystem('ijk')
    >>> output_cs = CoordinateSystem('xyz')
    >>> cm = Affine(np.identity(4), input_cs, output_cs)
    >>> print reorder_output(cm, 'xzy').output_coords
    name: '-reordered', coord_names: ('x', 'z', 'y'), coord_dtype: float64
    >>> print reorder_output(cm, [0,2,1]).output_coords.coord_names
    ('x', 'z', 'y')

    >>> newcm = reorder_output(cm, 'yzx')
    >>> newcm.output_coords.coord_names
    ('y', 'z', 'x')

    """

    ndim = coordmap.ndim[1]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [coordmap.output_coords.index(s) for s in order]

    newaxes = [coordmap.output_coords.coord_names[i] for i in order]
    newoutcoords = CoordinateSystem(newaxes, coordmap.output_coords.name + '-reordered', coordmap.output_coords.coord_dtype)
    
    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    perm = perm.astype(coordmap.output_coords.coord_dtype)
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
    >>> cmap.input_coords.coord_names
    ('i', 'k', 'j')
    >>> cmap.output_coords.coord_names
    ('x', 'z', 'y')
    >>> cmap.affine
    array([[ 2.,  0.,  0.,  0.],
           [ 0.,  4.,  0.,  0.],
           [ 0.,  0.,  3.,  0.],
           [ 0.,  0.,  0.,  1.]])

    """
    ndimin = [cmap.ndim[0] for cmap in cmaps]
    ndimin.insert(0,0)
    ndimin = tuple(np.cumsum(ndimin))

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

    incoords = coordsys_product(*[cmap.input_coords for cmap in cmaps])
    outcoords = coordsys_product(*[cmap.output_coords for cmap in cmaps])

    if not notaffine:

        affine = linearize(mapping, ndimin[-1], dtype=incoords.coord_dtype)
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
    >>> id1.input_coords.coord_names
    ('x',)
    >>> id2.input_coords.coord_names
    ('i',)
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
            cmap = CoordinateMap(forward, 
                                 cmap.input_coords, 
                                 m.output_coords, 
                                 inverse_mapping=backward)
        else:
            raise ValueError(
                'input and output coordinates do not match: '
                'input=%s, output=%s' % 
                (`m.input_coords.dtype`, `cmap.output_coords.dtype`))

    notaffine = filter(lambda cmap: not isinstance(cmap, Affine), cmaps)
    if not notaffine:
        affine = linearize(cmap, 
                           cmap.ndim[0], 
                           dtype=cmap.output_coords.coord_dtype)
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

    raise NotImplementedError('The method this function depends on' 
                              'no longer exists.')
    """
    concat = CoordinateMap.from_affine([concataxis], [concataxis], 
                                       Affine(np.identity(2)), (n,))
    return product(concat, coordmap)
    """


def linearize(mapping, ndimin, step=1, origin=None, dtype=None):
    """
    Given a Mapping of ndimin variables, return the linearization of
    mapping at origin based on a given step size in each coordinate
    axis.

    If not specified, origin defaults to np.zeros(ndimin, dtype=dtype).
    
    Parameters
    ----------
    mapping : callable
       A function to linearize
    ndimin : int
       Number of input dimensions to mapping
    step : scalar, optional
       step size over which to calculate linear components.  Default 1
    origin : None or array, optional
       Origin at which to linearize mapping.  If None, origin is
       ``np.zeros(ndimin)``
    dtype : None or np.dtype, optional
       dtype for return.  Default is None.  If ``dtype`` is None, and
       ``step`` is an ndarray, use ``step.dtype``.  Otherwise use
       np.float.

    Returns
    -------
    C : array 
       Linearization of mapping in homogeneous coordinates, i.e.  an
       array of size (ndimout+1, ndimin+1) where ndimout =
       mapping(origin).shape[0].
    """
    if dtype is None:
        try:
            dtype = step.dtype
        except AttributeError:
            dtype = np.float
    step = np.array(step, dtype=dtype)
    if origin is None:
        origin = np.zeros(ndimin, dtype)
    else:
        if origin.dtype != dtype:
            warnings.warn('origin.dtype != dtype in function linearize, using input dtype')
        origin = np.asarray(origin, dtype=dtype)
        if origin.shape != (ndimin,):
            raise ValueError('origin.shape != (%d,)' % ndimin)
    b = mapping(origin)

    origin = np.multiply.outer(np.ones(ndimin, dtype), origin)
    y1 = mapping(step*np.eye(ndimin, dtype=dtype) + origin)
    y0 = mapping(origin)

    ndimout = y1.shape[1]
    C = np.zeros((ndimout+1, ndimin+1), (y0/step).dtype)
    C[-1,-1] = 1
    C[:ndimout,-1] = b
    C[:ndimout,:ndimin] = (y1 - y0).T / step
    return C

