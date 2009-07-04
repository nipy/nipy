"""
CoordinateMaps map (transform) an image from a domain (input space) to a 
range (output space).

A CoordinateMap object contains all the details about a domain
CoordinateSystem, a range CoordinateSystem and the mappings between
them.  The *mapping* transforms an image from the domain coordinate
system to the range coordinate system.  And the *inverse_mapping*
performs the opposite transformation.  The *inverse_mapping* can be
specified explicity when creating the CoordinateMap or implicitly in the
case of an Affine CoordinateMap.

"""

"""
Matthew, Cindee thoughts

Should we change the order of input args to CoordinateMap to:

CoordinateMap(function_domain, function_range, mapping, inverse_mapping=None)
or keep as
CoordinateMap(mapping, function_domain, function_range, inverse_mapping=None)

Affine should be renamed to AffineMap, or AffineCoordMap, or something

We need to think about whether coordinate values should be N by 3
(where 3 is the number of coordinates in the coordinate system), or 3
by N.

3 by N makes more sense when we think of applying an affine matrix to the points.

We might think, that if we have a matrix ``arr`` with points in, then
``arr[0]`` should be the first point [x,y,z], rather then the first coordinate
value of all the points - N values of [x]

That's as far as we got.

"""


import warnings

import numpy as np

from nipy.utils.onetime import setattr_on_read
import nipy.core.transforms.affines as affines
from nipy.core.reference.coordinate_system import(CoordinateSystem, 
                                                          safe_dtype)
from nipy.core.reference.coordinate_system import product as coordsys_product

__docformat__ = 'restructuredtext'

class CoordinateMap(object):
    """A set of domain and range CoordinateSystems and a function between them.

    For example, the function may represent the mapping of a voxel
    (the domain of the function) to real space (the range).  
    The function may be an affine or non-affine
    transformation.

    Attributes
    ----------
    function_domain : :class:`CoordinateSystem`
        The input coordinate system.
    function_range : :class:`CoordinateSystem`
        The output coordinate system.
    function : callable
        A callable that maps the function_domain to the function_range.
    inverse_function : None or callable
        A callable that maps the function_range to the function_domain.
        Not all functions have an inverse, in which case
        inverse_function is None.
        
    Examples
    --------
    >>> function_domain = CoordinateSystem('ijk', 'voxels')
    >>> function_range = CoordinateSystem('xyz', 'world')
    >>> mni_orig = np.array([-90.0, -126.0, -72.0])
    >>> function = lambda x: x + mni_orig
    >>> inv_function = lambda x: x - mni_orig
    >>> cm = CoordinateMap(function, function_domain, function_range, inv_function)

    Map the first 3 voxel coordinates, along the x-axis, to mni space:

    >>> x = np.array([[0,0,0], [1,0,0], [2,0,0]])
    >>> cm.function(x)
    array([[ -90., -126.,  -72.],
           [ -89., -126.,  -72.],
           [ -88., -126.,  -72.]])

    >>> x = CoordinateSystem('x')
    >>> y = CoordinateSystem('y')
    >>> m = CoordinateMap(np.exp, x, y, np.log)
    >>> m
    CoordinateMap(
       function=<ufunc 'exp'>,
       function_domain=CoordinateSystem(coord_names=('x',), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('y',), name='', coord_dtype=float64),
       inverse_function=<ufunc 'log'>
      )
    >>> m.inverse()
    CoordinateMap(
       function=<ufunc 'log'>,
       function_domain=CoordinateSystem(coord_names=('y',), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x',), name='', coord_dtype=float64),
       inverse_function=<ufunc 'exp'>
      )
    >>> 

    """

    _doc = {}
    function = np.exp
    _doc['function'] = 'The function from function_domain to function_range.'


    function_domain = CoordinateSystem('x')
    _doc['function_domain'] = 'The domain of the function, a CoordinateSystem.'


    function_range = CoordinateSystem('y')
    _doc['function_range'] = 'The range of the function, a CoordinateSystem.'


    inverse_function = np.log
    _doc['inverse_function'] = 'The inverse function from function_range' + \
                               'to function_domain, if supplied.'

    ndims = (1,1)
    _doc['ndims'] = 'Number of dimensions of domain and range, respectively.'

    def __init__(self, function, 
                 function_domain, 
                 function_range, 
                 inverse_function=None):
        """Create a CoordinateMap given the function and its
        domain and range.

        Parameters
        ----------
        function : callable
           The function between function_domain and function_range.
        function_domain : :class:`CoordinateSystem`
           The input coordinate system
        function_range : :class:`CoordinateSystem`
           The output coordinate system
        inverse_function : None or callable, optional
           The optional inverse of function, with the intention being
           ``x = inverse_function(function(x))``.  If the function is
           affine and invertible, then this is true for all x.  The
           default is None

        Returns
        -------
        coordmap : CoordinateMap
        """
        # These attrs define the structure of the coordmap.

        self.function = function
        self.function_domain = function_domain
        self.function_range = function_range
        self.inverse_function = inverse_function
        self.ndims = (function_domain.ndim, function_range.ndim)

        if not callable(function):
            raise ValueError('The function must be callable.')
        if inverse_function is not None:
            if not callable(inverse_function):
                raise ValueError('The inverse_function must be callable.')
        self._checkfunction()

    # All attributes are read only

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise AttributeError('the value of %s has already been set and all attributes are read-only' % key)
        object.__setattr__(self, key, value)

    ###################################################################
    #
    # Properties
    #
    ###################################################################


    ###################################################################
    #
    # Methods
    #
    ###################################################################


    def inverse(self):
        """
        Return a new CoordinateMap with the functions reversed
        """
        if self.inverse_function is None:
            return None
        return CoordinateMap(self.inverse_function, 
                             self.function_range, 
                             self.function_domain, 
                             inverse_function=self.function)

        
    def __call__(self, x):
        """Return mapping evaluated at x

        Also, check x and the return value of self.function for
        compatiblity with function_domain
        and function_range coordinate systems respectively.

        Parameters
        ----------
        x : array-like
           Values in domain coordinate system space that will be mapped
           to the range coordinate system space, using
           ``self.mapping``
           
        Returns
        -------
        y : array
           Values in range coordinate system space

        Examples
        --------
        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> mapping = lambda x:x+1
        >>> inverse = lambda x:x-1
        >>> cm = CoordinateMap(mapping, input_cs, output_cs, inverse)
        >>> cm([2,3,4])
        array([3, 4, 5])
        >>> cmi = cm.inverse()
        >>> cmi([2,6,12])
        array([ 1,  5, 11])

        """

        x = np.asarray(x)
        in_vals = self.function_domain._checked_values(x)
        out_vals = self.function(in_vals)
        final_vals = self.function_range._checked_values(out_vals)

        # Try to set the shape reasonably for self.ndims[0] == 1
        if x.ndim == 1:
            return final_vals.reshape(-1)
        elif x.ndim == 0:
            return np.squeeze(final_vals)
        else:
            return final_vals

    def __copy__(self):
        """Create a copy of the coordmap.

        Returns
        -------
        coordmap : CoordinateMap

        """

        return CoordinateMap(self.function, 
                             self.function_domain,
                             self.function_range, 
                             inverse_function=self.inverse_function)

    def reordered_domain(self, order=None, name=''):
        """
        Create a new coordmap with the coordinates of function_domain reordered.
        Default behaviour is to reverse the order of the coordinates.

        Parameters
        ----------
        order: sequence
             Order to use, defaults to reverse. The elements
             can be integers, strings or 2-tuples of strings.
             If they are strings, they should be in 
             self.function_domain.coord_names.

        name: string, optional
             Name of new function_domain, defaults to self.function_domain.name.

        Returns:
        --------

        newcoordmap: `CoordinateMap`
             A new CoordinateMap with reversed function_domain.

        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = AffineTransform(np.identity(4), input_cs, output_cs)
        >>> print cm.reordered_domain('ikj', name='neworder').function_domain
        CoordinateSystem(coord_names=('i', 'k', 'j'), name='neworder', coord_dtype=float64)
        """

        name = name or self.function_domain.name

        ndim = self.ndims[0]
        if order is None:
            order = range(ndim)[::-1]
        elif type(order[0]) == type(''):
            order = [self.function_domain.index(s) for s in order]

        newaxes = [self.function_domain.coord_names[i] for i in order]
        newincoords = CoordinateSystem(newaxes, 
                                       name,
                                       coord_dtype=self.function_domain.coord_dtype)
        perm = np.zeros((ndim+1,)*2)
        perm[-1,-1] = 1.

        for i, j in enumerate(order):
            perm[j,i] = 1.

        perm = perm.astype(self.function_domain.coord_dtype)
        A = AffineTransform(perm, newincoords, self.function_domain)
        return compose(self, A)


    def renamed_domain(self, newnames, name=''):
        """
        Create a new coordmap with the coordinates of function_domain renamed.

        Inputs:
        -------
        newnames: dictionary

             A dictionary whose keys are in
             self.function_domain.coord_names
             and whose values are the new names.

        name: string, optional
             Name of new function_domain, defaults to self.function_domain.name.

        Returns:
        --------

        newcoordmap: `CoordinateMap`
             A new CoordinateMap with renamed function_domain.

        >>> affine_domain = CoordinateSystem('ijk')
        >>> affine_range = CoordinateSystem('xyz')
        >>> affine_matrix = np.identity(4)
        >>> affine_mapping = AffineTransform(affine_matrix, affine_domain, affine_range)

        >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','j':'slice'})
        >>> print new_affine_mapping.function_domain
        CoordinateSystem(coord_names=('phase', 'slice', 'freq'), name='', coord_dtype=float64)

        >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','l':'slice'})
        Traceback (most recent call last):
           ...
        ValueError: no domain coordinate named l

        >>> 

        """

        name = name or self.function_domain.name

        for n in newnames:
            if n not in self.function_domain.coord_names:
                raise ValueError('no domain coordinate named %s' % str(n))

        new_coord_names = []
        for n in self.function_domain.coord_names:
            if n in newnames:
                new_coord_names.append(newnames[n])
            else:
                new_coord_names.append(n)

        new_function_domain = CoordinateSystem(new_coord_names,
                                            name, 
                                            coord_dtype=self.function_domain.coord_dtype)
        

        ndim = self.ndims[0]
        ident_map = AffineTransform(np.identity(ndim+1),
                           new_function_domain,
                           self.function_domain)

        return compose(self, ident_map)


    def renamed_range(self, newnames, name=''):
        """
        Create a new coordmap with the coordinates of function_range renamed.

        Parameters
        ----------
        newnames: dictionary

             A dictionary whose keys are in
             self.function_range.coord_names
             and whose values are the new names.

        name: string, optional
             Name of new function_domain, defaults to self.function_range.name.

        Returns:
        --------

        newcoordmap: `CoordinateMap`
             A new CoordinateMap with the coordinates of function_range renamed.

        >>> affine_domain = CoordinateSystem('ijk')
        >>> affine_range = CoordinateSystem('xyz')
        >>> affine_matrix = np.identity(4)
        >>> affine_mapping = AffineTransform(affine_matrix, affine_domain, affine_range)

        >>> new_affine_mapping = affine_mapping.renamed_range({'x':'u'})
        >>> print new_affine_mapping.function_range
        CoordinateSystem(coord_names=('u', 'y', 'z'), name='', coord_dtype=float64)

        >>> new_affine_mapping = affine_mapping.renamed_range({'w':'u'})
        Traceback (most recent call last):
           ...
        ValueError: no range coordinate named w

        >>> 

        """

        name = name or self.function_range.name

        for n in newnames:
            if n not in self.function_range.coord_names:
                raise ValueError('no range coordinate named %s' % str(n))

        new_coord_names = []
        for n in self.function_range.coord_names:
            if n in newnames:
                new_coord_names.append(newnames[n])
            else:
                new_coord_names.append(n)

        new_function_range = CoordinateSystem(new_coord_names,
                                             name, 
                                             coord_dtype=self.function_range.coord_dtype)
        
        ndim = self.ndims[1]
        ident_map = AffineTransform(np.identity(ndim+1),
                           self.function_range,
                           new_function_range)

        return compose(ident_map, self)

    def reordered_range(self, order=None, name=''):
        """
        Create a new coordmap with the coordinates of function_range reordered.
        Defaults to reversing the coordinates of function_range.

        Parameters
        ----------

        order: sequence
             Order to use, defaults to reverse. The elements
             can be integers, strings or 2-tuples of strings.
             If they are strings, they should be in 
             self.function_range.coord_names.

        name: string, optional
             Name of new function_range, defaults to self.function_range.name.

        Returns:
        --------

        newcoordmap: `CoordinateMap`
             A new CoordinateMap with reversed function_range.

        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = AffineTransform(np.identity(4), input_cs, output_cs)
        >>> print cm.reordered_range('xzy', name='neworder').function_range
        CoordinateSystem(coord_names=('x', 'z', 'y'), name='neworder', coord_dtype=float64)
        >>> print cm.reordered_range([0,2,1]).function_range.coord_names
        ('x', 'z', 'y')

        >>> newcm = cm.reordered_range('yzx')
        >>> newcm.function_range.coord_names
        ('y', 'z', 'x')

        """

        name = name or self.function_range.name

        ndim = self.ndims[1]
        if order is None:
            order = range(ndim)[::-1]
        elif type(order[0]) == type(''):
            order = [self.function_range.index(s) for s in order]

        newaxes = [self.function_range.coord_names[i] for i in order]
        newoutcoords = CoordinateSystem(newaxes, name, 
                                        self.function_range.coord_dtype)

        perm = np.zeros((ndim+1,)*2)
        perm[-1,-1] = 1.

        for i, j in enumerate(order):
            perm[j,i] = 1.

        perm = perm.astype(self.function_range.coord_dtype)
        A = AffineTransform(perm.T, self.function_range, newoutcoords)
        return compose(A, self)

    ###################################################################
    #
    # Private methods
    #
    ###################################################################

    def __repr__(self):
        if not hasattr(self, "inverse_function"):
            return "CoordinateMap(\n   function=%s,\n   function_domain=%s,\n   function_range=%s\n  )" % (repr(self.function), self.function_domain, self.function_range)
        else:
            return "CoordinateMap(\n   function=%s,\n   function_domain=%s,\n   function_range=%s,\n   inverse_function=%s\n  )" % (repr(self.function), self.function_domain, self.function_range, repr(self.inverse_function))


    def _checkfunction(self):
        """Verify that the domain and range of self.function work
        can be used for calling self.function.

        We do this by passing something that should work, through __call__
        """
        inp = np.zeros((10, self.ndims[0]),
                       dtype=self.function_domain.coord_dtype)
        out = self(inp)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and (self.function == other.function)
                and (self.function_domain == 
                     other.function_domain)
                and (self.function_range == 
                     other.function_range)
                and (self.inverse_function ==
                     other.inverse_function))

    def __ne__(self, other):
        return not self.__eq__(other)

class AffineTransform(CoordinateMap):
    """
    A class representing an affine transformation from a 
    domain to a range.
    
    This class has an affine attribute, which is a matrix representing
    the affine transformation in homogeneous coordinates.  This matrix
    is used to evaluate the function, rather than having an explicit
    function.

    >>> inp_cs = CoordinateSystem('ijk')
    >>> out_cs = CoordinateSystem('xyz')
    >>> cm = AffineTransform(np.diag([1, 2, 3, 1]), inp_cs, out_cs)
    >>> cm
    AffineTransform(
       affine=array([[ 1.,  0.,  0.,  0.],
                     [ 0.,  2.,  0.,  0.],
                     [ 0.,  0.,  3.,  0.],
                     [ 0.,  0.,  0.,  1.]]),
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=float64)
    )

    >>> cm.affine
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  2.,  0.,  0.],
           [ 0.,  0.,  3.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> cm([1,1,1])
    array([ 1.,  2.,  3.])
    >>> icm = cm.inverse()
    >>> icm([1,2,3])
    array([ 1.,  1.,  1.])
    
    """

    _doc = {}
    affine = np.diag([3,4,5,1])
    _doc['affine'] = 'The matrix representing an affine transformation ' + \
                       'homogeneous form.'


    function_domain = CoordinateSystem('x')
    _doc['function_domain'] = 'The domain of the affine transformation, ' + \
                              'a CoordinateSystem.'


    function_range = CoordinateSystem('y')
    _doc['function_range'] = 'The range of the affine transformation, ' + \
                             'a CoordinateSystem.'

    ndims = (3,3)
    _doc['ndims'] = 'Number of dimensions of domain and range, respectively.'


    def __init__(self, affine, function_domain, function_range):
        """
        Return an CoordinateMap specified by an affine transformation
        in homogeneous coordinates.
        
        Parameters
        ----------
        affine : array-like
           affine homogenous coordinate matrix
        function_domain : :class:`CoordinateSystem`
           input coordinates
        function_range : :class:`CoordinateSystem`
           output coordinates

        Notes
        -----
        The dtype of the resulting matrix is determined by finding a
        safe typecast for the function_domain, function_range and affine.
        """
        dtype = safe_dtype(affine.dtype,
                           function_domain.coord_dtype,
                           function_range.coord_dtype)
        inaxes = function_domain.coord_names
        outaxes = function_range.coord_names

        self.function_domain = CoordinateSystem(inaxes,
                                                function_domain.name,
                                                dtype)

        self.function_range = CoordinateSystem(outaxes,
                                               function_range.name,
                                               dtype)

        self.ndims = (self.function_domain.ndim,
                      self.function_range.ndim)
        affine = np.asarray(affine, dtype=dtype)
        if affine.shape != (self.ndims[1]+1, self.ndims[0]+1):
            raise ValueError('coordinate lengths do not match '
                             'affine matrix shape')

        # Test that it is actually an affine mapping in homogeneous
        # form

        bottom_row = np.array([0]*self.ndims[0] + [1])
        if not np.all(affine[-1] == bottom_row):
            raise ValueError('the homogeneous transform should have bottom=' + \
                             'row %s' % repr(bottom_row))

        self.affine = affine

    ###################################################################
    #
    # Properties
    #
    ###################################################################

    @setattr_on_read
    def function(self):
        A, b = affines.to_matrix_vector(self.affine)
        def _function(x):
            value = np.dot(x, A.T)
            value += b
            return value
        return _function
    _doc['function'] = 'The function of the AffineTransform.'

    @setattr_on_read
    def inverse_function(self):
        """The inverse affine function from the AffineTransform."""
        inverse = self.inverse()
        if inverse is not None:
            return inverse.function
        raise AttributeError('There is no inverse function for this affine ' + 
                             'because the transformation is not invertible.')

    _doc['inverse function'] = 'The inverse affine function of the ' + \
                               'AffineTransform, when appropriate.'

    def inverse(self):
        """
        Return the inverse affine transform, when appropriate, or None.
        """
        try:
            return AffineTransform(np.linalg.inv(self.affine), 
                                   self.function_range, 
                                   self.function_domain)
        except np.linalg.linalg.LinAlgError:
            return None


    ###################################################################
    #
    # Helper constructors
    #
    ###################################################################


    @staticmethod
    def from_params(innames, outnames, params):
        """
        Create an `AffineTransform` instance from sequences of innames and outnames.

        Parameters
        ----------
        innames : ``tuple`` of ``string``
           The names of the axes of the domain.
        outnames : ``tuple`` of ``string``
           The names of the axes of the range.
        params : `AffineTransform`, `ndarray` or `(ndarray, ndarray)`
           An affine function between the domain and range.  
           This can be represented either by a single
           ndarray (which is interpreted as the representation of the
           function in homogeneous coordinates) or an (A,b) tuple.

        Returns
        -------
        aff : `AffineTransform` object instance
        
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

        function_domain = CoordinateSystem(innames, "domain")
        function_range = CoordinateSystem(outnames, 'range')
        return AffineTransform(params, function_domain, function_range)

    @staticmethod
    def from_start_step(innames, outnames, start, step):
        """
        Create an `AffineTransform` instance from sequences of names, start
        and step.

        Parameters
        ----------
        innames : ``tuple`` of ``string``
            The names of the axes of the domain.
        outnames : ``tuple`` of ``string``
            The names of the axes of the range.
        start : ``tuple`` of ``float``
            Start vector used in constructing affine transformation
        step : ``tuple`` of ``float``
            Step vector used in constructing affine transformation

        Returns
        -------
        cm : `CoordinateMap`

        Examples
        --------
        >>> cm = AffineTransform.from_start_step('ijk', 'xyz', [1, 2, 3], [4, 5, 6])
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
        return AffineTransform.from_params(innames, 
                                  outnames, 
                                  (np.diag(step), start))

    @staticmethod
    def identity(names):
        """
        Return an identity coordmap of the given shape.
        
        Parameters
        ----------
        names : ``tuple`` of ``string`` 
           Names of Axes in domain CoordinateSystem

        Returns
        -------
        cm : `CoordinateMap` 
           ``CoordinateMap`` with `CoordinateSystem` domain and an
           identity transform, with identical domain and range.

        Examples
        --------
        >>> cm = AffineTransform.identity('ijk')
        >>> cm.affine
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])
        >>> print cm.function_domain
        CoordinateSystem(coord_names=('i', 'j', 'k'), name='domain', coord_dtype=float64)
        >>> print cm.function_range
        CoordinateSystem(coord_names=('i', 'j', 'k'), name='range', coord_dtype=float64)
        """
        return AffineTransform.from_start_step(names, names, [0]*len(names),
                                      [1]*len(names))


    ###################################################################
    #
    # Private methods
    #
    ###################################################################

    def __copy__(self):
        """
        Create a copy of the coordmap.

        Returns
        -------
        cm : `CoordinateMap`

        Examples
        --------
        >>> import copy
        >>> cm = AffineTransform(np.eye(4), CoordinateSystem('ijk'), CoordinateSystem('xyz'))
        >>> cm_copy = copy.copy(cm)
        >>> cm is cm_copy
        False

        Note that the matrix (affine) is not a pointer to the
        same data, it's a full independent copy

        >>> cm.affine[0,0] = 2.0
        >>> cm_copy.affine[0,0]
        1.0
        """
        return AffineTransform(self.affine.copy(), self.function_domain,
                      self.function_range)


    def __repr__(self):
        return "AffineTransform(\n   affine=%s,\n   function_domain=%s,\n   function_range=%s\n)" % ('\n          '.join(repr(self.affine).split('\n')), 
         self.function_domain, self.function_range)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and np.all(self.affine == other.affine)
                and (self.function_domain == 
                     other.function_domain)
                and (self.function_range == 
                     other.function_range))



def product(*cmaps):
    """
    Return the "topological" product of two or more CoordinateMaps.

    Parameters
    ----------
    cmaps : sequence of CoordinateMaps

    Returns
    -------
    cmap : ``CoordinateMap``

    >>> inc1 = AffineTransform.from_params('i', 'x', np.diag([2,1]))
    >>> inc2 = AffineTransform.from_params('j', 'y', np.diag([3,1]))
    >>> inc3 = AffineTransform.from_params('k', 'z', np.diag([4,1]))

    >>> cmap = product(inc1, inc3, inc2)
    >>> cmap.function_domain.coord_names
    ('i', 'k', 'j')
    >>> cmap.function_range.coord_names
    ('x', 'z', 'y')
    >>> cmap.affine
    array([[ 2.,  0.,  0.,  0.],
           [ 0.,  4.,  0.,  0.],
           [ 0.,  0.,  3.,  0.],
           [ 0.,  0.,  0.,  1.]])

    """
    ndimin = [cmap.ndims[0] for cmap in cmaps]
    ndimin.insert(0,0)
    ndimin = tuple(np.cumsum(ndimin))

    def function(x):
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

    notaffine = filter(lambda x: not isinstance(x, AffineTransform), cmaps)

    incoords = coordsys_product(*[cmap.function_domain for cmap in cmaps])
    outcoords = coordsys_product(*[cmap.function_range for cmap in cmaps])

    if not notaffine:
        affine = linearize(function, ndimin[-1], dtype=incoords.coord_dtype)
        return AffineTransform(affine, incoords, outcoords)
    return CoordinateMap(function, incoords, outcoords)


def compose(*cmaps):
    """
    Return the composition of two or more CoordinateMaps.

    Parameters
    ----------
    cmaps : sequence of CoordinateMaps

    Returns
    -------
    cmap : ``CoordinateMap``
         The resulting CoordinateMap has function_domain == cmaps[-1].function_domain
         and function_range == cmaps[0].function_range

    >>> cmap = AffineTransform.from_params('i', 'x', np.diag([2.,1.]))
    >>> cmapi = cmap.inverse()
    >>> id1 = compose(cmap,cmapi)
    >>> print id1.affine
    [[ 1.  0.]
     [ 0.  1.]]

    >>> id2 = compose(cmapi,cmap)
    >>> id1.function_domain.coord_names
    ('x',)
    >>> id2.function_domain.coord_names
    ('i',)
    >>> 

    """

    def _compose2(cmap1, cmap2):
        forward = lambda input: cmap1.function(cmap2.function(input))
        cmap1i = cmap1.inverse()
        cmap2i = cmap2.inverse()
        if cmap1i is not None and cmap2i is not None:
            backward = lambda output: cmap2i.function(cmap1i.function(output))
        else:
            backward = None
        return forward, backward

    cmap = cmaps[-1]
    for i in range(len(cmaps)-2,-1,-1):
        m = cmaps[i]
        if m.function_domain == cmap.function_range:
            forward, backward = _compose2(m, cmap)
            cmap = CoordinateMap(forward, 
                                 cmap.function_domain, 
                                 m.function_range, 
                                 inverse_function=backward)
        else:
            raise ValueError(
                'domain and range coordinates do not match: '
                'domain=%s, range=%s' % 
                (`m.function_domain.dtype`, `cmap.function_range.dtype`))

    notaffine = filter(lambda cmap: not isinstance(cmap, AffineTransform), cmaps)
    if not notaffine:
        affine = linearize(cmap, 
                           cmap.ndims[0], 
                           dtype=cmap.function_range.coord_dtype)
        return AffineTransform(affine, cmap.function_domain,
                      cmap.function_range)
    return cmap
    

def concat(coordmap, axis_name='concat', append=False):
    """
    Create a CoordinateMap by adding prepending or appending a new
    coordinate named axis_name to both domain and range
    coordinates.

    Parameters
    ----------

    coordmap : CoordinateMap
       The coordmap to be used

    axis_name : str
       The name of the new dimension formed by concatenation

    append : bool
       If True, append the coordinate, else prepend it.

    >>> affine_domain = CoordinateSystem('ijk')
    >>> affine_range = CoordinateSystem('xyz')
    >>> affine_matrix = np.diag([3,4,5,1])
    >>> affine_mapping = AffineTransform(affine_matrix, affine_domain, affine_range)
    >>> concat(affine_mapping, 't')
    AffineTransform(
       affine=array([[ 1.,  0.,  0.,  0.,  0.],
                     [ 0.,  3.,  0.,  0.,  0.],
                     [ 0.,  0.,  4.,  0.,  0.],
                     [ 0.,  0.,  0.,  5.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]]),
       function_domain=CoordinateSystem(coord_names=('t', 'i', 'j', 'k'), name='product', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('t', 'x', 'y', 'z'), name='product', coord_dtype=float64)
    )

    >>> concat(affine_mapping, 't', append=True)
    AffineTransform(
       affine=array([[ 3.,  0.,  0.,  0.,  0.],
                     [ 0.,  4.,  0.,  0.,  0.],
                     [ 0.,  0.,  5.,  0.,  0.],
                     [ 0.,  0.,  0.,  1.,  0.],
                     [ 0.,  0.,  0.,  0.,  1.]]),
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 't'), name='product', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x', 'y', 'z', 't'), name='product', coord_dtype=float64)
    )
    >>> 

    """

    coords = CoordinateSystem([axis_name])
    concat = AffineTransform(np.identity(2),
                    coords,
                    coords)
    if not append:
        return product(concat, coordmap)
    else:
        return product(coordmap, concat)


def linearize(function, ndimin, step=1, origin=None, dtype=None):
    """
    Given a function of ndimin variables, return the linearization of
    function at origin based on a given step size in each coordinate
    axis.

    If not specified, origin defaults to np.zeros(ndimin, dtype=dtype).
    
    Parameters
    ----------
    function : callable
       A function to linearize
    ndimin : int
       Number of input dimensions to function
    step : scalar, optional
       step size over which to calculate linear components.  Default 1
    origin : None or array, optional
       Origin at which to linearize function.  If None, origin is
       ``np.zeros(ndimin)``
    dtype : None or np.dtype, optional
       dtype for return.  Default is None.  If ``dtype`` is None, and
       ``step`` is an ndarray, use ``step.dtype``.  Otherwise use
       np.float.

    Returns
    -------
    C : array 
       Linearization of function in homogeneous coordinates, i.e.  an
       array of size (ndimout+1, ndimin+1) where ndimout =
       function(origin).shape[0].
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
            warnings.warn('origin.dtype != dtype in function linearize, using domain dtype')
        origin = np.asarray(origin, dtype=dtype)
        if origin.shape != (ndimin,):
            raise ValueError('origin.shape != (%d,)' % ndimin)
    b = function(origin)

    origin = np.multiply.outer(np.ones(ndimin, dtype), origin)
    y1 = function(step*np.eye(ndimin, dtype=dtype) + origin)
    y0 = function(origin)

    ndimout = y1.shape[1]
    C = np.zeros((ndimout+1, ndimin+1), (y0/step).dtype)
    C[-1,-1] = 1
    C[:ndimout,-1] = b
    C[:ndimout,:ndimin] = (y1 - y0).T / step
    return C

