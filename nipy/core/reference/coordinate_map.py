"""
This module describes two types of *mappings*:

* CoordinateMap: a general function from a domain to a range, with a possible
     inverse function.

* AffineTransform: an affine function from a domain to a range, not
     necessarily of the same dimension, hence not always invertible.

Each of these objects is meant to encapsulate a tuple of 
(domain, range, function).
Each of the mapping objects contain all the details about their domain
CoordinateSystem, their range CoordinateSystem and the mapping between
them.  

Common API
----------

They are separate classes, neither one inheriting from the other.
They do, however, share some parts of an API, each having methods:

* renamed_domain : rename on the coordinates of the domain (returns a new mapping)

* renamed_range : rename the coordinates of the range (returns a new mapping)

* reordered_domain : reorder the coordinates of the domain (returns a new mapping)

* reordered_range : reorder the coordinates of the range (returns a new mapping)

* inverse : when appropriate, return the inverse *mapping*

These methods are implemented by module level functions of the same name.

They also share some attributes:

* ndims : the dimensions of the domain and range, respectively

* function_domain : CoordinateSystem describing the domain

* function_range : CoordinateSystem describing the range

Operations on mappings (module level functions)
-----------------------------------------------

* compose : Take a sequence of mappings (either CoordinateMaps or AffineTransforms)
   and return their composition. If they are all AffineTransforms, an AffineTransform is
   returned. This checks to ensure that domains and ranges of the various
   mappings agree.

* product : Take a sequence of mappings (either CoordinateMaps or AffineTransforms)
   and return a new mapping that has domain and range given by the concatenation of their 
   domains and ranges, and the mapping simply concatenates the output of
   each of the individual mappings. If they are all AffineTransforms, an AffineTransform is
   returned. If they are all AffineTransforms that are in fact linear (i.e. origin=0)
   then can is represented as a block matrix with the size of the blocks determined by

* concat : Take a mapping and prepend a coordinate to its domain and range. 
   For mapping `m`, this is the same as product(AffineTransform.identity('concat'), `m`)

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
    >>> cm = CoordinateMap(function_domain, function_range, function, inv_function)

    Map the first 3 voxel coordinates, along the x-axis, to mni space:

    >>> x = np.array([[0,0,0], [1,0,0], [2,0,0]])
    >>> cm.function(x)
    array([[ -90., -126.,  -72.],
           [ -89., -126.,  -72.],
           [ -88., -126.,  -72.]])

    >>> x = CoordinateSystem('x')
    >>> y = CoordinateSystem('y')
    >>> m = CoordinateMap(x, y, np.exp, np.log)
    >>> m
    CoordinateMap(
       function_domain=CoordinateSystem(coord_names=('x',), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('y',), name='', coord_dtype=float64),
       function=<ufunc 'exp'>,
       inverse_function=<ufunc 'log'>
      )
    >>> m.inverse()
    CoordinateMap(
       function_domain=CoordinateSystem(coord_names=('y',), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x',), name='', coord_dtype=float64),
       function=<ufunc 'log'>,
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

    def __init__(self, function_domain, 
                 function_range, 
                 function, 
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
        warnings.warn('CoordinateMaps are not as robust as AffineTransform')

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

    def reordered_domain(self, order=None, name=''):
        """
        Create a new CoordinateMap with the coordinates of function_domain reordered.
        Default behaviour is to reverse the order of the coordinates.

        Parameters
        ----------
        order: sequence
             Order to use, defaults to reverse. The elements
             can be integers, strings or 2-tuples of strings.
             If they are strings, they should be in 
             mapping.function_domain.coord_names.

        name: string, optional
             Name of new function_domain, defaults to mapping.function_domain.name.

        Returns:
        --------

        newmapping :CoordinateMap
             A new CoordinateMap with the coordinates of function_domain reordered.

        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = CoordinateMap(input_cs, output_cs, lambda x:x+1)
        >>> print cm.reordered_domain('ikj', name='neworder').function_domain
        CoordinateSystem(coord_names=('i', 'k', 'j'), name='neworder', coord_dtype=float64)
        """

        return reordered_domain(self, order, name)

    def reordered_range(self, order=None, name=''):
        """
        Create a new CoordinateMap with the coordinates of function_range reordered.
        Defaults to reversing the coordinates of function_range.

        Parameters
        ----------

        order: sequence
             Order to use, defaults to reverse. The elements
             can be integers, strings or 2-tuples of strings.
             If they are strings, they should be in 
             mapping.function_range.coord_names.

        name: string, optional
             Name of new function_range, defaults to mapping.function_range.name.

        Returns:
        --------

        newmapping : CoordinateMap
             A new CoordinateMap with the coordinates of function_range reordered.

        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = CoordinateMap(input_cs, output_cs, lambda x:x+1)
        >>> print cm.reordered_range('xzy', name='neworder').function_range
        CoordinateSystem(coord_names=('x', 'z', 'y'), name='neworder', coord_dtype=float64)
        >>> print cm.reordered_range([0,2,1]).function_range.coord_names
        ('x', 'z', 'y')

        >>> newcm = cm.reordered_range('yzx')
        >>> newcm.function_range.coord_names
        ('y', 'z', 'x')

        """

        return reordered_range(self, order, name)

    def renamed_domain(self, newnames, name=''):
        """
        Create a new CoordinateMap with the coordinates of function_domain renamed.

        Inputs:
        -------
        newnames: dictionary

             A dictionary whose keys are in
             mapping.function_domain.coord_names
             and whose values are the new names.

        name: string, optional
             Name of new function_domain, defaults to mapping.function_domain.name.

        Returns:
        --------

        newmaping : CoordinateMap
             A new CoordinateMap with renamed function_domain. 

        >>> domain = CoordinateSystem('ijk')
        >>> range = CoordinateSystem('xyz')
        >>> cm = CoordinateMap(domain, range, lambda x:x+1)

        >>> new_cm = cm.renamed_domain({'i':'phase','k':'freq','j':'slice'})
        >>> print new_cm.function_domain
        CoordinateSystem(coord_names=('phase', 'slice', 'freq'), name='', coord_dtype=float64)

        >>> new_cm = cm.renamed_domain({'i':'phase','k':'freq','l':'slice'})
        Traceback (most recent call last):
           ...
        ValueError: no domain coordinate named l

        >>> 

        """

        return renamed_domain(self, newnames, name)

    def renamed_range(self, newnames, name=''):
        """
        Create a new CoordinateMap with the coordinates of function_domain renamed.

        Inputs:
        -------
        newnames: dictionary

             A dictionary whose keys are in
             mapping.function_domain.coord_names
             and whose values are the new names.

        name: string, optional
             Name of new function_domain, defaults to mapping.function_domain.name.

        Returns:
        --------

        newmapping : CoordinateMap
             A new CoordinateMap with renamed function_domain. 

        >>> domain = CoordinateSystem('ijk')
        >>> range = CoordinateSystem('xyz')
        >>> cm = CoordinateMap(domain, range, lambda x:x+1)

        >>> new_cm = cm.renamed_range({'x':'u'})
        >>> print new_cm.function_range
        CoordinateSystem(coord_names=('u', 'y', 'z'), name='', coord_dtype=float64)

        >>> new_cm = cm.renamed_range({'w':'u'})
        Traceback (most recent call last):
           ...
        ValueError: no range coordinate named w


        """

        return renamed_range(self, newnames, name)

    def inverse(self):
        """
        Return a new CoordinateMap with the functions reversed
        """
        if self.inverse_function is None:
            return None
        return CoordinateMap(self.function_range, 
                             self.function_domain, 
                             self.inverse_function,
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
        >>> function = lambda x:x+1
        >>> inverse = lambda x:x-1
        >>> cm = CoordinateMap(input_cs, output_cs, function, inverse)
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

        return CoordinateMap(self.function_domain,
                             self.function_range, 
                             self.function, 
                             inverse_function=self.inverse_function)

    ###################################################################
    #
    # Private methods
    #
    ###################################################################

    def __repr__(self):
        if not hasattr(self, "inverse_function"):
            return "CoordinateMap(\n   function_domain=%s,\n   function_range=%s,\n   function=%s\n  )" % (self.function_domain, self.function_range, repr(self.function))
        else:
            return "CoordinateMap(\n   function_domain=%s,\n   function_range=%s,\n   function=%s,\n   inverse_function=%s\n  )" % (self.function_domain, self.function_range, repr(self.function), repr(self.inverse_function))


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


class AffineTransform(object):
    """
    A class representing an affine transformation from a 
    domain to a range.
    
    This class has an affine attribute, which is a matrix representing
    the affine transformation in homogeneous coordinates.  This matrix
    is used to evaluate the function, rather than having an explicit
    function.

    >>> inp_cs = CoordinateSystem('ijk')
    >>> out_cs = CoordinateSystem('xyz')
    >>> cm = AffineTransform(inp_cs, out_cs, np.diag([1, 2, 3, 1]))
    >>> cm
    AffineTransform(
       function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float64),
       function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='', coord_dtype=float64),
       affine=array([[ 1.,  0.,  0.,  0.],
                     [ 0.,  2.,  0.,  0.],
                     [ 0.,  0.,  3.,  0.],
                     [ 0.,  0.,  0.,  1.]])
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


    def __init__(self, function_domain, function_range, affine):
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

    def inverse(self):
        """
        Return the inverse affine transform, when appropriate, or None.
        """
        try:
            return AffineTransform(self.function_range, 
                                   self.function_domain,
                                   np.linalg.inv(self.affine))
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
        return AffineTransform(function_domain, function_range, params)

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
    # Methods
    #
    ###################################################################

    def reordered_domain(self, order=None, name=''):
        """
        Create a new AffineTransform with the coordinates of function_domain reordered.
        Default behaviour is to reverse the order of the coordinates.

        Parameters
        ----------
        order: sequence
             Order to use, defaults to reverse. The elements
             can be integers, strings or 2-tuples of strings.
             If they are strings, they should be in 
             mapping.function_domain.coord_names.

        name: string, optional
             Name of new function_domain, defaults to mapping.function_domain.name.

        Returns:
        --------

        newmapping :AffineTransform
             A new AffineTransform with the coordinates of function_domain reordered.

        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = AffineTransform(input_cs, output_cs, np.identity(4))
        >>> print cm.reordered_domain('ikj', name='neworder').function_domain
        CoordinateSystem(coord_names=('i', 'k', 'j'), name='neworder', coord_dtype=float64)
        """

        return reordered_domain(self, order, name)

    def reordered_range(self, order=None, name=''):
        """
        Create a new AffineTransform with the coordinates of function_range reordered.
        Defaults to reversing the coordinates of function_range.

        Parameters
        ----------

        order: sequence
             Order to use, defaults to reverse. The elements
             can be integers, strings or 2-tuples of strings.
             If they are strings, they should be in 
             mapping.function_range.coord_names.

        name: string, optional
             Name of new function_range, defaults to mapping.function_range.name.

        Returns:
        --------

        newmapping : AffineTransform
             A new AffineTransform with the coordinates of function_range reordered.

        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = AffineTransform(input_cs, output_cs, np.identity(4))
        >>> print cm.reordered_range('xzy', name='neworder').function_range
        CoordinateSystem(coord_names=('x', 'z', 'y'), name='neworder', coord_dtype=float64)
        >>> print cm.reordered_range([0,2,1]).function_range.coord_names
        ('x', 'z', 'y')

        >>> newcm = cm.reordered_range('yzx')
        >>> newcm.function_range.coord_names
        ('y', 'z', 'x')

        """

        return reordered_range(self, order, name)

    def renamed_domain(self, newnames, name=''):
        """
        Create a new AffineTransform with the coordinates of function_domain renamed.

        Inputs:
        -------
        newnames: dictionary

             A dictionary whose keys are in
             mapping.function_domain.coord_names
             and whose values are the new names.

        name: string, optional
             Name of new function_domain, defaults to mapping.function_domain.name.

        Returns:
        --------

        newmapping : AffineTransform
             A new AffineTransform with renamed function_domain. 

        >>> affine_domain = CoordinateSystem('ijk')
        >>> affine_range = CoordinateSystem('xyz')
        >>> affine_matrix = np.identity(4)
        >>> affine_mapping = AffineTransform(affine_domain, affine_range, affine_matrix)

        >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','j':'slice'})
        >>> print new_affine_mapping.function_domain
        CoordinateSystem(coord_names=('phase', 'slice', 'freq'), name='', coord_dtype=float64)

        >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','l':'slice'})
        Traceback (most recent call last):
           ...
        ValueError: no domain coordinate named l

        >>> 

        """

        return renamed_domain(self, newnames, name)

    def renamed_range(self, newnames, name=''):
        """
        Create a new AffineTransform with the coordinates of function_domain renamed.

        Inputs:
        -------
        newnames: dictionary

             A dictionary whose keys are in
             mapping.function_domain.coord_names
             and whose values are the new names.

        name: string, optional
             Name of new function_domain, defaults to mapping.function_domain.name.

        Returns:
        --------

        newmapping : AffineTransform
             A new AffineTransform with renamed function_domain. 

        >>> affine_domain = CoordinateSystem('ijk')
        >>> affine_range = CoordinateSystem('xyz')
        >>> affine_matrix = np.identity(4)
        >>> affine_mapping = AffineTransform(affine_domain, affine_range, affine_matrix)

        >>> new_affine_mapping = affine_mapping.renamed_range({'x':'u'})
        >>> print new_affine_mapping.function_range
        CoordinateSystem(coord_names=('u', 'y', 'z'), name='', coord_dtype=float64)

        >>> new_affine_mapping = affine_mapping.renamed_range({'w':'u'})
        Traceback (most recent call last):
           ...
        ValueError: no range coordinate named w


        """

        return renamed_range(self, newnames, name)
    
    def __call__(self, x):
        """Return mapping evaluated at x

        Parameters
        ----------
        x : array-like
           Values in domain coordinate system space that will be mapped
           to the range coordinate system space, using
           the homogeneous transform matrix self.affine.
           
        Returns
        -------
        y : array
           Values in range coordinate system space

        Examples
        --------
        >>> input_cs = CoordinateSystem('ijk', coord_dtype=np.int)
        >>> output_cs = CoordinateSystem('xyz', coord_dtype=np.int)
        >>> affine = np.array([[1,0,0,1],
        ...                    [0,1,0,1],
        ...                    [0,0,1,1],
        ...                    [0,0,0,1]])
        >>> affine_transform = AffineTransform(input_cs, output_cs, affine)
        >>> affine_transform([2,3,4])
        array([3, 4, 5])
        >>> affine_transform_inv = affine_transform.inverse()
        >>> # Its inverse has a matrix of np.float
        >>> # because np.linalg.inv was called.
        >>> affine_transform_inv([2,6,12])
        array([  1.,   5.,  11.])

        """

        x = np.asarray(x)
        A, b = affines.to_matrix_vector(self.affine)
        x_reshaped = x.reshape((-1, self.ndims[0]))
        y_reshaped = np.dot(x_reshaped, A.T) + b[np.newaxis,:]
        y = y_reshaped.reshape(x.shape[:-1] + (self.ndims[1],))
        return y

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
        >>> cm = AffineTransform(CoordinateSystem('ijk'), CoordinateSystem('xyz'), np.eye(4))
        >>> cm_copy = copy.copy(cm)
        >>> cm is cm_copy
        False

        Note that the matrix (affine) is not a pointer to the
        same data, it's a full independent copy

        >>> cm.affine[0,0] = 2.0
        >>> cm_copy.affine[0,0]
        1.0
        """
        return AffineTransform(self.function_domain,
                               self.function_range,
                               self.affine.copy())


    def __repr__(self):
        return "AffineTransform(\n   function_domain=%s,\n   function_range=%s,\n   affine=%s\n)" % (self.function_domain, 
                                                                                                     self.function_range,
                                                                                                     '\n          '.join(repr(self.affine).split('\n')))

    def __eq__(self, other):
        test1, test2, test3, test4 =(isinstance(other, self.__class__), 
                                     np.allclose(self.affine, other.affine), 
                                     (self.function_domain == 
                                      other.function_domain),
                                     (self.function_range == 
                                      other.function_range))
        value = test1 and test2 and test3 and test4
        return value

    def __ne__(self, other):
        return not self.__eq__(other)

####################################################################################
#
# Module level functions
#
####################################################################################

def product(*cmaps):
    """
    Return the "topological" product of two or more mappings, which
    can be either AffineTransforms or CoordinateMaps.

    If they are all AffineTransforms, the result is an AffineTransform,
    else it is a CoordinateMap.

    Parameters
    ----------
    cmaps : sequence of CoordinateMaps or AffineTransforms

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

    >>> A1 = AffineTransform.from_params('ij', 'xyz', np.array([[2,3,1,0],[3,4,5,0],[7,9,3,1]]).T)
    >>> A2 = AffineTransform.from_params('xyz', 'de', np.array([[8,6,7,4],[1,-1,13,3],[0,0,0,1]]))

    >>> A1.affine
    array([[ 2.,  3.,  7.],
           [ 3.,  4.,  9.],
           [ 1.,  5.,  3.],
           [ 0.,  0.,  1.]])
    >>> A2.affine
    array([[  8.,   6.,   7.,   4.],
           [  1.,  -1.,  13.,   3.],
           [  0.,   0.,   0.,   1.]])

    >>> p=product(A1, A2)
    >>> p.affine
    array([[  2.,   3.,   0.,   0.,   0.,   7.],
           [  3.,   4.,   0.,   0.,   0.,   9.],
           [  1.,   5.,   0.,   0.,   0.,   3.],
           [  0.,   0.,   8.,   6.,   7.,   4.],
           [  0.,   0.,   1.,  -1.,  13.,   3.],
           [  0.,   0.,   0.,   0.,   0.,   1.]])

    >>> print np.allclose(p.affine[:3,:2], A1.affine[:3,:2])
    True
    >>> print np.allclose(p.affine[:3,-1], A1.affine[:3,-1])
    True
    >>> print np.allclose(p.affine[3:5,2:5], A2.affine[:2,:3])
    True
    >>> print np.allclose(p.affine[3:5,-1], A2.affine[:2,-1])
    True
    >>> 

    >>> A1([3,4])
    array([ 25.,  34.,  26.])
    >>> A2([5,6,7])
    array([ 129.,   93.])
    >>> p([3,4,5,6,7])
    array([  25.,   34.,   26.,  129.,   93.])

    """

    # First, check if they're all Affine
    allaffine = np.all([isinstance(cmap, AffineTransform) for cmap in cmaps])
    if allaffine:
        return _product_affines(*cmaps)
    else:
        warnings.warn("product of non-affine CoordinateMaps is less robust than"+
                      "the AffineTransform")
        return _product_cmaps(*[_as_coordinate_map(cmap) for cmap in cmaps])

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

    # First check if they're all affine

    allaffine = np.all([isinstance(cmap, AffineTransform) for cmap in cmaps])
    if allaffine:
        return _compose_affines(*cmaps)
    else:
        warnings.warn("composition of non-affine CoordinateMaps is less robust than"+
                      "the AffineTransform")
        return _compose_cmaps(*[_as_coordinate_map(cmap) for cmap in cmaps])



def reordered_domain(mapping, order=None, name=''):
    """
    Create a new coordmap with the coordinates of function_domain reordered.
    Default behaviour is to reverse the order of the coordinates.

    Parameters
    ----------
    order: sequence
         Order to use, defaults to reverse. The elements
         can be integers, strings or 2-tuples of strings.
         If they are strings, they should be in 
         mapping.function_domain.coord_names.

    name: string, optional
         Name of new function_domain, defaults to mapping.function_domain.name.

    Returns:
    --------

    newmapping : CoordinateMap or AffineTransform
         A new CoordinateMap with the coordinates of function_domain reordered.
         If isinstance(mapping, AffineTransform), newmapping is also
         an AffineTransform. Otherwise, it is a CoordinateMap.

    >>> input_cs = CoordinateSystem('ijk')
    >>> output_cs = CoordinateSystem('xyz')
    >>> cm = AffineTransform(input_cs, output_cs, np.identity(4))
    >>> print cm.reordered_domain('ikj', name='neworder').function_domain
    CoordinateSystem(coord_names=('i', 'k', 'j'), name='neworder', coord_dtype=float64)
    """

    name = name or mapping.function_domain.name

    ndim = mapping.ndims[0]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [mapping.function_domain.index(s) for s in order]

    newaxes = [mapping.function_domain.coord_names[i] for i in order]
    newincoords = CoordinateSystem(newaxes, 
                                   name,
                                   coord_dtype=mapping.function_domain.coord_dtype)
    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    perm = perm.astype(mapping.function_domain.coord_dtype)
    A = AffineTransform(newincoords, mapping.function_domain, perm)

    if isinstance(mapping, AffineTransform):
        return _compose_affines(mapping, A)
    else:
        return _compose_cmaps(mapping, _as_coordinate_map(A))


def renamed_domain(mapping, newnames, name=''):
    """
    Create a new coordmap with the coordinates of function_domain renamed.

    Inputs:
    -------
    newnames: dictionary

         A dictionary whose keys are in
         mapping.function_domain.coord_names
         and whose values are the new names.

    name: string, optional
         Name of new function_domain, defaults to mapping.function_domain.name.

    Returns:
    --------

    newmapping : CoordinateMap or AffineTransform
         A new mapping with renamed function_domain. If 
         isinstance(mapping, AffineTransform), newmapping is also
         an AffineTransform. Otherwise, it is a CoordinateMap.

    >>> affine_domain = CoordinateSystem('ijk')
    >>> affine_range = CoordinateSystem('xyz')
    >>> affine_matrix = np.identity(4)
    >>> affine_mapping = AffineTransform(affine_domain, affine_range, affine_matrix)

    >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','j':'slice'})
    >>> print new_affine_mapping.function_domain
    CoordinateSystem(coord_names=('phase', 'slice', 'freq'), name='', coord_dtype=float64)

    >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','l':'slice'})
    Traceback (most recent call last):
       ...
    ValueError: no domain coordinate named l

    >>> 

    """

    name = name or mapping.function_domain.name

    for n in newnames:
        if n not in mapping.function_domain.coord_names:
            raise ValueError('no domain coordinate named %s' % str(n))

    new_coord_names = []
    for n in mapping.function_domain.coord_names:
        if n in newnames:
            new_coord_names.append(newnames[n])
        else:
            new_coord_names.append(n)

    new_function_domain = CoordinateSystem(new_coord_names,
                                           name, 
                                           coord_dtype=mapping.function_domain.coord_dtype)


    ndim = mapping.ndims[0]
    ident_map = AffineTransform(new_function_domain,
                                mapping.function_domain, 
                                np.identity(ndim+1))

    if isinstance(mapping, AffineTransform):
        return _compose_affines(mapping, ident_map)
    else:
        return _compose_cmaps(mapping, _as_coordinate_map(ident_map))


def renamed_range(mapping, newnames, name=''):
    """
    Create a new coordmap with the coordinates of function_range renamed.

    Parameters
    ----------
    newnames: dictionary

         A dictionary whose keys are in
         mapping.function_range.coord_names
         and whose values are the new names.

    name: string, optional
         Name of new function_domain, defaults to mapping.function_range.name.

    Returns:
    --------

    newmapping : CoordinateMap or AffineTransform
         A new CoordinateMap with the coordinates of function_range renamed.
         If isinstance(mapping, AffineTransform), newmapping is also
         an AffineTransform. Otherwise, it is a CoordinateMap.

    >>> affine_domain = CoordinateSystem('ijk')
    >>> affine_range = CoordinateSystem('xyz')
    >>> affine_matrix = np.identity(4)
    >>> affine_mapping = AffineTransform(affine_domain, affine_range, affine_matrix)

    >>> new_affine_mapping = affine_mapping.renamed_range({'x':'u'})
    >>> print new_affine_mapping.function_range
    CoordinateSystem(coord_names=('u', 'y', 'z'), name='', coord_dtype=float64)

    >>> new_affine_mapping = affine_mapping.renamed_range({'w':'u'})
    Traceback (most recent call last):
       ...
    ValueError: no range coordinate named w

    >>> 

    """

    name = name or mapping.function_range.name

    for n in newnames:
        if n not in mapping.function_range.coord_names:
            raise ValueError('no range coordinate named %s' % str(n))

    new_coord_names = []
    for n in mapping.function_range.coord_names:
        if n in newnames:
            new_coord_names.append(newnames[n])
        else:
            new_coord_names.append(n)

    new_function_range = CoordinateSystem(new_coord_names,
                                         name, 
                                         coord_dtype=mapping.function_range.coord_dtype)

    ndim = mapping.ndims[1]
    ident_map = AffineTransform(mapping.function_range,
                                new_function_range,
                                np.identity(ndim+1))

    if isinstance(mapping, AffineTransform):
        return _compose_affines(ident_map, mapping)
    else:
        return _compose_cmaps(_as_coordinate_map(ident_map), mapping)


def reordered_range(mapping, order=None, name=''):
    """
    Create a new coordmap with the coordinates of function_range reordered.
    Defaults to reversing the coordinates of function_range.

    Parameters
    ----------

    order: sequence
         Order to use, defaults to reverse. The elements
         can be integers, strings or 2-tuples of strings.
         If they are strings, they should be in 
         mapping.function_range.coord_names.

    name: string, optional
         Name of new function_range, defaults to mapping.function_range.name.

    Returns:
    --------

    newmapping : CoordinateMap or AffineTransform
         A new CoordinateMap with the coordinates of function_range reordered.
         If isinstance(mapping, AffineTransform), newmapping is also
         an AffineTransform. Otherwise, it is a CoordinateMap.

    >>> input_cs = CoordinateSystem('ijk')
    >>> output_cs = CoordinateSystem('xyz')
    >>> cm = AffineTransform(input_cs, output_cs, np.identity(4))
    >>> print cm.reordered_range('xzy', name='neworder').function_range
    CoordinateSystem(coord_names=('x', 'z', 'y'), name='neworder', coord_dtype=float64)
    >>> print cm.reordered_range([0,2,1]).function_range.coord_names
    ('x', 'z', 'y')

    >>> newcm = cm.reordered_range('yzx')
    >>> newcm.function_range.coord_names
    ('y', 'z', 'x')

    """

    name = name or mapping.function_range.name

    ndim = mapping.ndims[1]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [mapping.function_range.index(s) for s in order]

    newaxes = [mapping.function_range.coord_names[i] for i in order]
    newoutcoords = CoordinateSystem(newaxes, name, 
                                    mapping.function_range.coord_dtype)

    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    perm = perm.astype(mapping.function_range.coord_dtype)
    A = AffineTransform(mapping.function_range, newoutcoords, perm.T)

    if isinstance(mapping, AffineTransform):
        return _compose_affines(A, mapping)
    else:
        return _compose_cmaps(_as_coordinate_map(A), mapping)


###################################################################
#
# Private functions
#
###################################################################

def _as_coordinate_map(cmap):
    """
    Take a mapping AffineTransform and return a
    CoordinateMap with the appropriate functions.
    """

    if isinstance(cmap, CoordinateMap):
        return cmap
    elif isinstance(cmap, AffineTransform):
        affine_transform = cmap
        A, b = affines.to_matrix_vector(affine_transform.affine)

        def _function(x):
            value = np.dot(x, A.T)
            value += b
            return value

        affine_transform_inv = affine_transform.inverse()
        if affine_transform_inv:
            Ainv, binv = affines.to_matrix_vector(affine_transform_inv.affine)
            def _inverse_function(x):
                value = np.dot(x, Ainv.T)
                value += binv
                return value
        else:
            _inverse_function = None

        return CoordinateMap(affine_transform.function_domain,
                             affine_transform.function_range,
                             _function,
                             _inverse_function)
    else:
        raise ValueError('all mappings should be instances of either CoordinateMap or AffineTransform')

def _compose_affines(*affines):
    """
    Return the composition of a sequence of affines,
    checking the domains and ranges.
    """

    cur = AffineTransform(affines[-1].function_domain,
                          affines[-1].function_domain,
                          np.identity(affines[-1].ndims[0]+1,
                                      dtype=affines[-1].affine.dtype))
    for cmap in affines[::-1]:
        if cmap.function_domain == cur.function_range:
            cur = AffineTransform(cur.function_domain,
                                  cmap.function_range,
                                  np.dot(cmap.affine, cur.affine))
        else:
            raise ValueError("domains and ranges don't match up correctly")
    return cur

def _compose_cmaps(*cmaps):
    """
    Compute the composition of a sequence of cmaps
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

    # the identity coordmap
    cur = CoordinateMap(cmaps[-1].function_domain,
                        cmaps[-1].function_domain,
                        lambda x: x,
                        lambda x: x)
    for cmap in cmaps[::-1]:
        if cmap.function_domain == cur.function_range:
            forward, backward = _compose2(cmap, cur)
            cur =  CoordinateMap(cur.function_domain, 
                                 cmap.function_range, 
                                 forward,
                                 inverse_function=backward)
        else:
            raise ValueError(
                'domain and range coordinates do not match: '
                'domain=%s, range=%s' % 
                (`cmap.function_domain.dtype`, `cur.function_range.dtype`))

    return cur

def _product_cmaps(*cmaps):
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

    return CoordinateMap(incoords, outcoords, function)


def _product_affines(*affine_mappings):
    """
    Product of affine_mappings.
    """

    ndimin = [affine.ndims[0] for affine in affine_mappings]
    ndimout = [affine.ndims[1] for affine in affine_mappings]

    M = np.zeros((np.sum(ndimout)+1, np.sum(ndimin)+1),
                 dtype=safe_dtype(*[affine.affine.dtype for affine in affine_mappings]))
    M[-1,-1] = 1.

    # Fill in the block matrix
    product_domain = []
    product_range = []

    i = 0
    j = 0

    for l, affine in enumerate(affine_mappings):
        A, b = affines.to_matrix_vector(affine.affine)
        M[i:(i+ndimout[l]),j:(j+ndimin[l])] = A
        M[i:(i+ndimout[l]),-1] = b
        product_domain.extend(affine.function_domain.coord_names)
        product_range.extend(affine.function_range.coord_names)
        i += ndimout[l]
        j += ndimin[l]

    return AffineTransform(
        CoordinateSystem(product_domain, name='product', coord_dtype=M.dtype), 
        CoordinateSystem(product_range, name='product', coord_dtype=M.dtype), 
        M)

