# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

* compose : Take a sequence of mappings (either CoordinateMaps or
   AffineTransforms) and return their composition. If they are all
   AffineTransforms, an AffineTransform is returned. This checks to
   ensure that domains and ranges of the various mappings agree.
* product : Take a sequence of mappings (either CoordinateMaps or
   AffineTransforms) and return a new mapping that has domain and range
   given by the concatenation of their domains and ranges, and the
   mapping simply concatenates the output of each of the individual
   mappings. If they are all AffineTransforms, an AffineTransform is
   returned. If they are all AffineTransforms that are in fact linear
   (i.e. origin=0) then can is represented as a block matrix with the
   size of the blocks determined by
* concat : Take a mapping and prepend a coordinate to its domain and
   range.  For mapping `m`, this is the same as
   product(AffineTransform.identity('concat'), `m`)
"""

import warnings

import numpy as np

from nibabel.orientations import io_orientation

from ..transforms.affines import from_matrix_vector, to_matrix_vector
from .coordinate_system import(CoordinateSystem,
                               safe_dtype,
                               product as coordsys_product
                               )

# shorthand
CS = CoordinateSystem


class CoordinateMap(object):
    """A set of domain and range CoordinateSystems and a function between them.

    For example, the function may represent the mapping of a voxel (the
    domain of the function) to real space (the range).  The function may
    be an affine or non-affine transformation.

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
       Not all functions have an inverse, in which case inverse_function
       is None.

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
        """Create a CoordinateMap given function, domain and range.

        Parameters
        ----------
        function_domain : :class:`CoordinateSystem`
           The input coordinate system.
        function_range : :class:`CoordinateSystem`
           The output coordinate system
        function : callable
           The function between function_domain and function_range.  It
           should be a callable that accepts arrays of shape (N,
           function_domain.ndim) and returns arrays of shape (N,
           function_range.ndim), where N is the number of points for
           transformation.
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
        self.function_domain = function_domain
        self.function_range = function_range
        self.function = function
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
            raise AttributeError('the value of %s has already been '
                                 'set and all attributes are '
                                 'read-only' % key)
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

    def reordered_domain(self, order=None):
        """
        Create a new CoordinateMap with the coordinates of function_domain reordered.
        Default behaviour is to reverse the order of the coordinates.

        Parameters
        ----------
        order : sequence
           Order to use, defaults to reverse. The elements can be
           integers, strings or 2-tuples of strings.  If they are
           strings, they should be in
           mapping.function_domain.coord_names.

        Returns
        -------
        newmapping : CoordinateMap
           A new CoordinateMap with the coordinates of function_domain
           reordered.

        Examples
        --------
        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = CoordinateMap(input_cs, output_cs, lambda x:x+1)
        >>> cm.reordered_domain('ikj').function_domain
        CoordinateSystem(coord_names=('i', 'k', 'j'), name='', coord_dtype=float64)
        """
        return reordered_domain(self, order)

    def reordered_range(self, order=None):
        """ Nnew CoordinateMap with function_range reordered.

        Defaults to reversing the coordinates of function_range.

        Parameters
        ----------
        order : sequence
           Order to use, defaults to reverse. The elements can be
           integers, strings or 2-tuples of strings.  If they are
           strings, they should be in
           mapping.function_range.coord_names.

        Returns
        -------
        newmapping : CoordinateMap
           A new CoordinateMap with the coordinates of function_range
           reordered.

        Examples
        --------
        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = CoordinateMap(input_cs, output_cs, lambda x:x+1)
        >>> cm.reordered_range('xzy').function_range
        CoordinateSystem(coord_names=('x', 'z', 'y'), name='', coord_dtype=float64)
        >>> cm.reordered_range([0,2,1]).function_range.coord_names
        ('x', 'z', 'y')

        >>> newcm = cm.reordered_range('yzx')
        >>> newcm.function_range.coord_names
        ('y', 'z', 'x')
        """
        return reordered_range(self, order)

    def renamed_domain(self, newnames, name=''):
        """ New CoordinateMap with function_domain renamed

        Parameters
        ----------
        newnames : dict
           A dictionary whose keys are integers or are in
           mapping.function_domain.coord_names and whose values are the
           new names.

        Returns
        -------
        newmaping : CoordinateMap
           A new CoordinateMap with renamed function_domain.

        Examples
        --------
        >>> domain = CoordinateSystem('ijk')
        >>> range = CoordinateSystem('xyz')
        >>> cm = CoordinateMap(domain, range, lambda x:x+1)

        >>> new_cm = cm.renamed_domain({'i':'phase','k':'freq','j':'slice'})
        >>> new_cm.function_domain
        CoordinateSystem(coord_names=('phase', 'slice', 'freq'), name='', coord_dtype=float64)

        >>> new_cm = cm.renamed_domain({'i':'phase','k':'freq','l':'slice'})
        Traceback (most recent call last):
           ...
        ValueError: no domain coordinate named l
        """
        return renamed_domain(self, newnames)

    def renamed_range(self, newnames, name=''):
        """ New CoordinateMap with function_domain renamed

        Parameters
        ----------
        newnames : dict
           A dictionary whose keys are integers or are in
           mapping.function_range.coord_names and whose values are the
           new names.

        Returns
        -------
        newmapping : CoordinateMap
           A new CoordinateMap with renamed function_range.

        Examples
        --------
        >>> domain = CoordinateSystem('ijk')
        >>> range = CoordinateSystem('xyz')
        >>> cm = CoordinateMap(domain, range, lambda x:x+1)

        >>> new_cm = cm.renamed_range({'x':'u'})
        >>> new_cm.function_range
        CoordinateSystem(coord_names=('u', 'y', 'z'), name='', coord_dtype=float64)

        >>> new_cm = cm.renamed_range({'w':'u'})
        Traceback (most recent call last):
           ...
        ValueError: no range coordinate named w
        """
        return renamed_range(self, newnames)

    def inverse(self):
        """ New CoordinateMap with the functions reversed
        """
        if self.inverse_function is None:
            return None
        return CoordinateMap(self.function_range,
                             self.function_domain,
                             self.inverse_function,
                             inverse_function=self.function)

    def __call__(self, x):
        """ Return mapping evaluated at x

        Also, check x and the return value of self.function for
        compatiblity with function_domain and function_range coordinate
        systems respectively.

        Parameters
        ----------
        x : array-like
           Values in domain coordinate system space that will be mapped
           to the range coordinate system space, using ``self.mapping``.
           The last dimension of the array is the coordinate dimension.
           Thus `x` can be any array that can be reshaped to (N,
           self.function_domain.ndim), and that matches
           self.function_domain dtype.

        Returns
        -------
        y : array
           Values in range coordinate system space.  If input `x` was
           shape S + (self.function_domain.ndim) (where S is a tuple of
           int and can be ()) - then the output `y` will be shape S +
           (self.function_range.ndim)

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
        x = np.asanyarray(x)
        out_shape = (self.function_range.ndim,)
        if x.ndim > 1:
            out_shape = x.shape[:-1] + out_shape
        in_vals = self.function_domain._checked_values(x)
        out_vals = self.function(in_vals)
        final_vals = self.function_range._checked_values(out_vals)
        return final_vals.reshape(out_shape)

    def __copy__(self):
        """ Create a copy of the coordmap.

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
        if self.inverse_function is None:
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
    """ Class for affine transformation from domain to a range

    This class has an affine attribute, which is a matrix representing
    the affine transformation in homogeneous coordinates.  This matrix
    is used to evaluate the function, rather than having an explicit
    function (as is the case for a CoordinateMap).

    Examples
    --------
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
        """ Initialize AffineTransform

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
        affine = np.asarray(affine)
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
        """ Return inverse affine transform or None if not invertible
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
    def from_params(innames, outnames, params, domain_name='',
                    range_name=''):
        """ Create `AffineTransform` from `innames` and `outnames`

        Parameters
        ----------
        innames : sequence of str or str
           The names of the axes of the domain.  If str, then names
           given by ``list(innames)``
        outnames : seqence of str or str
           The names of the axes of the range. If str, then names
           given by ``list(outnames)``
        params : AffineTransform, array or (array, array)
           An affine function between the domain and range.
           This can be represented either by a single
           ndarray (which is interpreted as the representation of the
           function in homogeneous coordinates) or an (A,b) tuple.
        domain_name : str, optional
           Name of domain CoordinateSystem
        range_name : str, optional
           Name of range CoordinateSystem

        Returns
        -------
        aff : ``AffineTransform``

        Notes
        -----
        :Precondition: ``len(shape) == len(names)``
        :Raises ValueError: ``if len(shape) != len(names)``
        """
        if type(params) == type(()):
            A, b = params
            params = from_matrix_vector(A, b)
        ndim = (len(innames) + 1, len(outnames) + 1)
        if params.shape != ndim[::-1]:
            raise ValueError('shape and number of axis names do not agree')
        function_domain = CoordinateSystem(innames, domain_name)
        function_range = CoordinateSystem(outnames, range_name)
        return AffineTransform(function_domain, function_range, params)

    @staticmethod
    def from_start_step(innames, outnames, start, step, domain_name='',
                        range_name=''):
        """ New `AffineTransform` from names, start and step.

        Parameters
        ----------
        innames : sequence of str or str
           The names of the axes of the domain.  If str, then names
           given by ``list(innames)``
        outnames : seqence of str or str
           The names of the axes of the range. If str, then names
           given by ``list(outnames)``
        start : sequence of float
           Start vector used in constructing affine transformation
        step : sequence of float
           Step vector used in constructing affine transformation
        domain_name : str, optional
           Name of domain CoordinateSystem
        range_name : str, optional
           Name of range CoordinateSystem

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
                                           (np.diag(step), start),
                                           domain_name=domain_name,
                                           range_name=range_name)

    @staticmethod
    def identity(coord_names, name=''):
        """ Return an identity coordmap of the given shape

        Parameters
        ----------
        coord_names : sequence of str or str
           The names of the axes of the domain.  If str, then names
           given by ``list(coord_names)``
        name : str, optional
           Name of origin of coordinate system

        Returns
        -------
        cm : `CoordinateMap`
           ``CoordinateMap`` with ``CoordinateSystem`` domain and an
           identity transform, with identical domain and range.

        Examples
        --------
        >>> cm = AffineTransform.identity('ijk', 'somewhere')
        >>> cm.affine
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])
        >>> cm.function_domain
        CoordinateSystem(coord_names=('i', 'j', 'k'), name='somewhere', coord_dtype=float64)
        >>> cm.function_range
        CoordinateSystem(coord_names=('i', 'j', 'k'), name='somewhere', coord_dtype=float64)
        """
        return AffineTransform.from_start_step(coord_names, coord_names, [0]*len(coord_names),
                                      [1]*len(coord_names), name, name)


    ###################################################################
    #
    # Methods
    #
    ###################################################################

    def reordered_domain(self, order=None):
        """ New AffineTransform with function_domain reordered

        Default behaviour is to reverse the order of the coordinates.

        Parameters
        ----------
        order : sequence
           Order to use, defaults to reverse. The elements can be
           integers, strings or 2-tuples of strings.  If they are
           strings, they should be in
           mapping.function_domain.coord_names.

        Returns
        -------
        newmapping :AffineTransform
           A new AffineTransform with the coordinates of function_domain
           reordered.

        Examples
        --------
        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = AffineTransform(input_cs, output_cs, np.identity(4))
        >>> cm.reordered_domain('ikj').function_domain
        CoordinateSystem(coord_names=('i', 'k', 'j'), name='', coord_dtype=float64)
        """

        return reordered_domain(self, order)

    def reordered_range(self, order=None):
        """ New AffineTransform with function_range reordered

        Defaults to reversing the coordinates of function_range.

        Parameters
        ----------
        order : sequence
           Order to use, defaults to reverse. The elements can be
           integers, strings or 2-tuples of strings.  If they are
           strings, they should be in
           mapping.function_range.coord_names.

        Returns
        -------
        newmapping : AffineTransform
           A new AffineTransform with the coordinates of function_range
           reordered.

        Examples
        --------
        >>> input_cs = CoordinateSystem('ijk')
        >>> output_cs = CoordinateSystem('xyz')
        >>> cm = AffineTransform(input_cs, output_cs, np.identity(4))
        >>> cm.reordered_range('xzy').function_range
        CoordinateSystem(coord_names=('x', 'z', 'y'), name='', coord_dtype=float64)
        >>> cm.reordered_range([0,2,1]).function_range.coord_names
        ('x', 'z', 'y')

        >>> newcm = cm.reordered_range('yzx')
        >>> newcm.function_range.coord_names
        ('y', 'z', 'x')
        """
        return reordered_range(self, order)

    def renamed_domain(self, newnames, name=''):
        """ New AffineTransform with function_domain renamed

        Parameters
        ----------
        newnames : dict
           A dictionary whose keys are integers or are in
           mapping.function_domain.coord_names and whose values are the
           new names.

        Returns
        -------
        newmapping : AffineTransform
           A new AffineTransform with renamed function_domain.

        Examples
        --------
        >>> affine_domain = CoordinateSystem('ijk')
        >>> affine_range = CoordinateSystem('xyz')
        >>> affine_matrix = np.identity(4)
        >>> affine_mapping = AffineTransform(affine_domain, affine_range, affine_matrix)

        >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','j':'slice'})
        >>> new_affine_mapping.function_domain
        CoordinateSystem(coord_names=('phase', 'slice', 'freq'), name='', coord_dtype=float64)

        >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','l':'slice'})
        Traceback (most recent call last):
           ...
        ValueError: no domain coordinate named l
        """
        return renamed_domain(self, newnames)

    def renamed_range(self, newnames, name=''):
        """ New AffineTransform with renamed function_domain

        Parameters
        ----------
        newnames : dict
           A dictionary whose keys are integers or are in
           mapping.function_range.coord_names and whose values are the
           new names.

        Returns
        -------
        newmapping : AffineTransform
           A new AffineTransform with renamed function_range.

        Examples
        --------
        >>> affine_domain = CoordinateSystem('ijk')
        >>> affine_range = CoordinateSystem('xyz')
        >>> affine_matrix = np.identity(4)
        >>> affine_mapping = AffineTransform(affine_domain, affine_range, affine_matrix)

        >>> new_affine_mapping = affine_mapping.renamed_range({'x':'u'})
        >>> new_affine_mapping.function_range
        CoordinateSystem(coord_names=('u', 'y', 'z'), name='', coord_dtype=float64)

        >>> new_affine_mapping = affine_mapping.renamed_range({'w':'u'})
        Traceback (most recent call last):
           ...
        ValueError: no range coordinate named w
        """
        return renamed_range(self, newnames)

    def __call__(self, x):
        """Return mapping evaluated at x

        Parameters
        ----------
        x : array-like
           Values in domain coordinate system space that will be mapped
           to the range coordinate system space, using the homogeneous
           transform matrix self.affine.  The last dimension of the
           array is the coordinate dimension.  Thus `x` can be any array
           that can be reshaped to (N, self.function_domain.ndim), and
           that matches self.function_domain dtype.


        Returns
        -------
        y : array
           Values in range coordinate system space. If input `x` was
           shape S + (self.function_domain.ndim) (where S is a tuple of
           int and can be ()) - then the output `y` will be shape S +
           (self.function_range.ndim)

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
        x = np.asanyarray(x)
        out_shape = (self.function_range.ndim,)
        if x.ndim > 1:
            out_shape = x.shape[:-1] + out_shape
        in_vals = self.function_domain._checked_values(x)
        A, b = to_matrix_vector(self.affine)
        out_vals = np.dot(in_vals, A.T) + b[np.newaxis,:]
        final_vals = self.function_range._checked_values(out_vals)
        return final_vals.reshape(out_shape)

    ###################################################################
    #
    # Private methods
    #
    ###################################################################

    def __copy__(self):
        """ Create a copy of the AffineTransform.

        Returns
        -------
        affine_transform : AffineTransform

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
    """ "topological" product of two or more mappings

    The mappings can be either AffineTransforms or CoordinateMaps.

    If they are all AffineTransforms, the result is an AffineTransform,
    else it is a CoordinateMap.

    Parameters
    ----------
    cmaps : sequence of CoordinateMaps or AffineTransforms

    Returns
    -------
    cmap : ``CoordinateMap``

    Examples
    --------
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

    >>> np.allclose(p.affine[:3,:2], A1.affine[:3,:2])
    True
    >>> np.allclose(p.affine[:3,-1], A1.affine[:3,-1])
    True
    >>> np.allclose(p.affine[3:5,2:5], A2.affine[:2,:3])
    True
    >>> np.allclose(p.affine[3:5,-1], A2.affine[:2,-1])
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
    """ Return the composition of two or more CoordinateMaps.

    Parameters
    ----------
    cmaps : sequence of CoordinateMaps

    Returns
    -------
    cmap : ``CoordinateMap``
       The resulting CoordinateMap has function_domain ==
       cmaps[-1].function_domain and function_range ==
       cmaps[0].function_range

    Examples
    --------
    >>> cmap = AffineTransform.from_params('i', 'x', np.diag([2.,1.]))
    >>> cmapi = cmap.inverse()
    >>> id1 = compose(cmap,cmapi)
    >>> id1.affine
    array([[ 1.,  0.],
           [ 0.,  1.]])

    >>> id2 = compose(cmapi,cmap)
    >>> id1.function_domain.coord_names
    ('x',)
    >>> id2.function_domain.coord_names
    ('i',)
    """
    # First check if they're all affine
    allaffine = np.all([isinstance(cmap, AffineTransform) for cmap in cmaps])
    if allaffine:
        return _compose_affines(*cmaps)
    else:
        warnings.warn("composition of non-affine CoordinateMaps is "
                      "less robust than the AffineTransform")
        return _compose_cmaps(*[_as_coordinate_map(cmap) for cmap in cmaps])


def reordered_domain(mapping, order=None):
    """ New coordmap with the coordinates of function_domain reordered

    Default behaviour is to reverse the order of the coordinates.

    Parameters
    ----------
    order: sequence
       Order to use, defaults to reverse. The elements can be integers,
       strings or 2-tuples of strings.  If they are strings, they should
       be in mapping.function_domain.coord_names.

    Returns
    -------
    newmapping : CoordinateMap or AffineTransform
       A new CoordinateMap with the coordinates of function_domain
       reordered.  If isinstance(mapping, AffineTransform), newmapping
       is also an AffineTransform. Otherwise, it is a CoordinateMap.

    Examples
    --------
    >>> input_cs = CoordinateSystem('ijk')
    >>> output_cs = CoordinateSystem('xyz')
    >>> cm = AffineTransform(input_cs, output_cs, np.identity(4))
    >>> cm.reordered_domain('ikj').function_domain
    CoordinateSystem(coord_names=('i', 'k', 'j'), name='', coord_dtype=float64)

    Notes
    -----
    If no reordering is to be performed, it returns a copy of mapping.
    """
    ndim = mapping.ndims[0]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [mapping.function_domain.index(s) for s in order]

    newaxes = [mapping.function_domain.coord_names[i] for i in order]
    newincoords = CoordinateSystem(newaxes,
                                   mapping.function_domain.name,
                                   coord_dtype=mapping.function_domain.coord_dtype)
    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    perm = perm.astype(mapping.function_domain.coord_dtype)

    # If there is no reordering, return mapping
    if np.allclose(perm, np.identity(perm.shape[0])):
        import copy
        return copy.copy(mapping)

    A = AffineTransform(newincoords, mapping.function_domain, perm)
    if isinstance(mapping, AffineTransform):
        return _compose_affines(mapping, A)
    else:
        return _compose_cmaps(mapping, _as_coordinate_map(A))


def shifted_domain_origin(mapping, difference_vector, new_origin):
    """ Shift the origin of the domain

    Parameters
    ----------
    difference_vector : array
       Representing the difference shifted_origin-current_origin in the
       domain's basis.

    Examples
    --------
    >>> A = np.random.standard_normal((5,6))
    >>> A[-1] = [0,0,0,0,0,1]
    >>> affine_transform = AffineTransform(CS('ijklm', 'oldorigin'), CS('xyzt'), A)
    >>> affine_transform.function_domain
    CoordinateSystem(coord_names=('i', 'j', 'k', 'l', 'm'), name='oldorigin', coord_dtype=float64)

    A random change of origin

    >>> difference = np.random.standard_normal(5)

    The same affine transforation with a different origin for its domain

    >>> shifted_affine_transform = shifted_domain_origin(affine_transform, difference, 'neworigin')
    >>> shifted_affine_transform.function_domain
    CoordinateSystem(coord_names=('i', 'j', 'k', 'l', 'm'), name='neworigin', coord_dtype=float64)

    Let's check that things work

    >>> point_in_old_basis = np.random.standard_normal(5)

    This is the relation ship between coordinates in old and new origins

    >>> np.allclose(shifted_affine_transform(point_in_old_basis), affine_transform(point_in_old_basis+difference))
    True
    >>> np.allclose(shifted_affine_transform(point_in_old_basis-difference), affine_transform(point_in_old_basis))
    True
    """
    new_function_domain = CoordinateSystem(mapping.function_domain.coord_names,
                                           new_origin,
                                           coord_dtype=mapping.function_domain.coord_dtype)

    ndim = new_function_domain.ndim
    shift_matrix = np.identity(ndim+1,
                               mapping.function_domain.coord_dtype)
    shift_matrix[:-1,-1] = np.array(difference_vector)
    shift_map = AffineTransform(new_function_domain,
                                mapping.function_domain,
                                shift_matrix)

    if isinstance(mapping, AffineTransform):
        return _compose_affines(mapping, shift_map)
    else:
        return _compose_cmaps(mapping, _as_coordinate_map(shift_map))


def shifted_range_origin(mapping, difference_vector, new_origin):
    """ Shift the origin of the range.

    Parameters
    ----------
    difference_vector : array
       Representing the difference shifted_origin-current_origin in the
       range's basis.

    Examples
    --------
    >>> A = np.random.standard_normal((5,6))
    >>> A[-1] = [0,0,0,0,0,1]
    >>> affine_transform = AffineTransform(CS('ijklm'), CS('xyzt', 'oldorigin'), A)
    >>> affine_transform.function_range
    CoordinateSystem(coord_names=('x', 'y', 'z', 't'), name='oldorigin', coord_dtype=float64)

    Make a random shift of the origin in the range

    >>> difference = np.random.standard_normal(4)
    >>> shifted_affine_transform = shifted_range_origin(affine_transform, difference, 'neworigin')
    >>> shifted_affine_transform.function_range
    CoordinateSystem(coord_names=('x', 'y', 'z', 't'), name='neworigin', coord_dtype=float64)
    >>>

    Evaluate the transform and verify it does as expected

    >>> point_in_domain = np.random.standard_normal(5)

    Check that things work

    >>> np.allclose(shifted_affine_transform(point_in_domain), affine_transform(point_in_domain) - difference)
    True
    >>> np.allclose(shifted_affine_transform(point_in_domain) + difference, affine_transform(point_in_domain))
    True
    """
    new_function_range = CoordinateSystem(mapping.function_range.coord_names,
                                          new_origin,
                                          coord_dtype=mapping.function_range.coord_dtype)

    ndim = new_function_range.ndim
    shift_matrix = np.identity(ndim+1,
                               mapping.function_range.coord_dtype)
    shift_matrix[:-1,-1] = -np.array(difference_vector)
    shift_map = AffineTransform(mapping.function_range,
                                new_function_range,
                                shift_matrix)

    if isinstance(mapping, AffineTransform):
        return _compose_affines(shift_map, mapping)
    else:
        return _compose_cmaps(_as_coordinate_map(shift_map), mapping)


def renamed_domain(mapping, newnames, name=''):
    """ New coordmap with the coordinates of function_domain renamed

    Parameters
    ----------
    newnames: dict
       A dictionary whose keys are integers or are in
       mapping.function_range.coord_names and whose values are the new
       names.

    Returns
    -------
    newmapping : CoordinateMap or AffineTransform
       A new mapping with renamed function_domain. If
       isinstance(mapping, AffineTransform), newmapping is also an
       AffineTransform. Otherwise, it is a CoordinateMap.

    Examples
    --------
    >>> affine_domain = CoordinateSystem('ijk')
    >>> affine_range = CoordinateSystem('xyz')
    >>> affine_matrix = np.identity(4)
    >>> affine_mapping = AffineTransform(affine_domain, affine_range, affine_matrix)

    >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','j':'slice'})
    >>> new_affine_mapping.function_domain
    CoordinateSystem(coord_names=('phase', 'slice', 'freq'), name='', coord_dtype=float64)

    >>> new_affine_mapping = affine_mapping.renamed_domain({'i':'phase','k':'freq','l':'slice'})
    Traceback (most recent call last):
       ...
    ValueError: no domain coordinate named l
    """
    for key in newnames.keys():
        if type(key) == type(0):
            newnames[mapping.function_domain.coord_names[key]] = \
                newnames[key]
            del(newnames[key])

    for key in newnames.keys():
        if key not in mapping.function_domain.coord_names:
            raise ValueError('no domain coordinate named %s' % str(key))

    new_coord_names = []
    for n in mapping.function_domain.coord_names:
        if n in newnames:
            new_coord_names.append(newnames[n])
        else:
            new_coord_names.append(n)

    new_function_domain = CoordinateSystem(new_coord_names,
                                           mapping.function_domain.name,
                                           coord_dtype=mapping.function_domain.coord_dtype)

    ndim = mapping.ndims[0]
    ident_map = AffineTransform(new_function_domain,
                                mapping.function_domain,
                                np.identity(ndim+1))

    if isinstance(mapping, AffineTransform):
        return _compose_affines(mapping, ident_map)
    else:
        return _compose_cmaps(mapping, _as_coordinate_map(ident_map))


def renamed_range(mapping, newnames):
    """ New coordmap with the coordinates of function_range renamed

    Parameters
    ----------
    newnames : dict
       A dictionary whose keys are integers or in
       mapping.function_range.coord_names and whose values are the new
       names.

    Returns
    -------
    newmapping : CoordinateMap or AffineTransform
       A new CoordinateMap with the coordinates of function_range
       renamed.  If isinstance(mapping, AffineTransform), newmapping is
       also an AffineTransform. Otherwise, it is a CoordinateMap.

    Examples
    --------
    >>> affine_domain = CoordinateSystem('ijk')
    >>> affine_range = CoordinateSystem('xyz')
    >>> affine_matrix = np.identity(4)
    >>> affine_mapping = AffineTransform(affine_domain, affine_range, affine_matrix)
    >>> new_affine_mapping = affine_mapping.renamed_range({'x':'u'})
    >>> new_affine_mapping.function_range
    CoordinateSystem(coord_names=('u', 'y', 'z'), name='', coord_dtype=float64)

    >>> new_affine_mapping = affine_mapping.renamed_range({'w':'u'})
    Traceback (most recent call last):
       ...
    ValueError: no range coordinate named w
    """
    for key in newnames.keys():
        if type(key) == type(0):
            newnames[mapping.function_range.coord_names[key]] = \
                newnames[key]
            del(newnames[key])

    for key in newnames.keys():
        if key not in mapping.function_range.coord_names:
            raise ValueError('no range coordinate named %s' % str(key))

    new_coord_names = []
    for n in mapping.function_range.coord_names:
        if n in newnames:
            new_coord_names.append(newnames[n])
        else:
            new_coord_names.append(n)

    new_function_range = CoordinateSystem(new_coord_names,
                                          mapping.function_range.name,
                                          coord_dtype=mapping.function_range.coord_dtype)

    ndim = mapping.ndims[1]
    ident_map = AffineTransform(mapping.function_range,
                                new_function_range,
                                np.identity(ndim+1))

    if isinstance(mapping, AffineTransform):
        return _compose_affines(ident_map, mapping)
    else:
        return _compose_cmaps(_as_coordinate_map(ident_map), mapping)


def reordered_range(mapping, order=None):
    """ New coordmap with the coordinates of function_range reordered

    Defaults to reversing the coordinates of function_range.

    Parameters
    ----------
    order: sequence
       Order to use, defaults to reverse. The elements can be integers,
       strings or 2-tuples of strings.  If they are strings, they should
       be in mapping.function_range.coord_names.

    Returns
    -------
    newmapping : CoordinateMap or AffineTransform
       A new CoordinateMap with the coordinates of function_range
       reordered.  If isinstance(mapping, AffineTransform), newmapping
       is also an AffineTransform. Otherwise, it is a CoordinateMap.

    Examples
    --------
    >>> input_cs = CoordinateSystem('ijk')
    >>> output_cs = CoordinateSystem('xyz')
    >>> cm = AffineTransform(input_cs, output_cs, np.identity(4))
    >>> cm.reordered_range('xzy').function_range
    CoordinateSystem(coord_names=('x', 'z', 'y'), name='', coord_dtype=float64)
    >>> cm.reordered_range([0,2,1]).function_range.coord_names
    ('x', 'z', 'y')

    >>> newcm = cm.reordered_range('yzx')
    >>> newcm.function_range.coord_names
    ('y', 'z', 'x')

    Notes
    -----
    If no reordering is to be performed, it returns a copy of mapping.
    """
    ndim = mapping.ndims[1]
    if order is None:
        order = range(ndim)[::-1]
    elif type(order[0]) == type(''):
        order = [mapping.function_range.index(s) for s in order]

    newaxes = [mapping.function_range.coord_names[i] for i in order]
    newoutcoords = CoordinateSystem(newaxes, mapping.function_range.name,
                                    mapping.function_range.coord_dtype)

    perm = np.zeros((ndim+1,)*2)
    perm[-1,-1] = 1.

    for i, j in enumerate(order):
        perm[j,i] = 1.

    perm = perm.astype(mapping.function_range.coord_dtype)
    if np.allclose(perm, np.identity(perm.shape[0])):
        import copy
        return copy.copy(mapping)


    A = AffineTransform(mapping.function_range, newoutcoords, perm.T)

    if isinstance(mapping, AffineTransform):
        return _compose_affines(A, mapping)
    else:
        return _compose_cmaps(_as_coordinate_map(A), mapping)


def equivalent(mapping1, mapping2):
    """
    A test to see if mapping1 is equal
    to mapping2 after possibly reordering the
    domain and range of mapping.

    Parameters
    ----------
    mapping1 : CoordinateMap or AffineTransform
    mapping2 : CoordinateMap or AffineTransform

    Returns
    -------
    are_they_equal : bool

    Examples
    --------
    >>> ijk = CoordinateSystem('ijk')
    >>> xyz = CoordinateSystem('xyz')
    >>> T = np.random.standard_normal((4,4))
    >>> T[-1] = [0,0,0,1] # otherwise AffineTransform raises
    ...                   # an exception because
    ...                   # it's supposed to represent an
    ...                   # affine transform in homogeneous
    ...                   # coordinates
    >>> A = AffineTransform(ijk, xyz, T)
    >>> B = A.reordered_domain('ikj').reordered_range('xzy')
    >>> C = B.renamed_domain({'i':'slice'})
    >>> equivalent(A, B)
    True
    >>> equivalent(A, C)
    False
    >>> equivalent(B, C)
    False
    >>>
    >>> D = CoordinateMap(ijk, xyz, np.exp)
    >>> equivalent(D, D)
    True
    >>> E = D.reordered_domain('kij').reordered_range('xzy')
    >>> # no non-AffineTransform will ever be
    >>> # equivalent to a reordered version of itself,
    >>> # because their functions don't evaluate as equal
    >>> equivalent(D, E)
    False
    >>> equivalent(E, E)
    True
    >>>
    >>> # This has not changed the order
    >>> # of the axes, so the function is still the same
    >>>
    >>> F = D.reordered_range('xyz').reordered_domain('ijk')
    >>> equivalent(F, D)
    True
    >>> id(F) == id(D)
    False
    """
    target_dnames = mapping2.function_domain.coord_names
    target_rnames = mapping2.function_range.coord_names

    try:
        mapping1 = mapping1.reordered_domain(target_dnames)\
            .reordered_range(target_rnames)
    except ValueError:
        # impossible to rename the domain and ranges of mapping1 to match mapping2
        return False

    return mapping1 == mapping2

###################################################################
#
# Private functions
#
###################################################################

def _as_coordinate_map(cmap):
    """ Return CoordinateMap from AffineTransform

    Take a mapping AffineTransform and return a
    CoordinateMap with the appropriate functions.
    """
    if isinstance(cmap, CoordinateMap):
        return cmap
    elif isinstance(cmap, AffineTransform):
        affine_transform = cmap
        A, b = to_matrix_vector(affine_transform.affine)

        def _function(x):
            value = np.dot(x, A.T)
            value += b
            return value

        affine_transform_inv = affine_transform.inverse()
        if affine_transform_inv:
            Ainv, binv = to_matrix_vector(affine_transform_inv.affine)
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
        raise ValueError('all mappings should be instances of '
                         'either CoordinateMap or AffineTransform')


def _compose_affines(*affines):
    """ Composition of sequence of affines

    Compose hecking the domains and ranges.
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
    """ Compute the composition of a sequence of cmaps
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
        x = np.atleast_2d(x)
        y = []
        for i in range(len(ndimin)-1):
            cmap = cmaps[i]
            yy = cmaps[i](x[:,ndimin[i]:ndimin[i+1]])
            y.append(yy)
        yy = np.hstack(y)
        return yy

    notaffine = filter(lambda x: not isinstance(x, AffineTransform), cmaps)

    incoords = coordsys_product(*[cmap.function_domain for cmap in cmaps])
    outcoords = coordsys_product(*[cmap.function_range for cmap in cmaps])

    return CoordinateMap(incoords, outcoords, function)


def _product_affines(*affine_mappings):
    """ Product of affine_mappings.
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
        A, b = to_matrix_vector(affine.affine)
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


def drop_io_dim(cm, name):
    ''' Drop dimension from coordinate map, if orthogonal to others

    Drops both input and corresponding output dimension.  Thus there
    should be a corresponding input dimension for a selected output
    dimension, or a corresponding output dimension for an input
    dimension.

    Parameters
    ----------
    cm : Affine
       Affine coordinate map instance
    name : str
       Name of input or output dimension to drop.  If this is an input
       dimension, there must be a corresponding output dimension with
       the same index, and vica versa.  The check for orthogonality
       ensures that the input and output dimensions are only related to
       each other, and not to the other dimensions.

    Returns
    -------
    cm_redux : Affine
       Affine coordinate map with orthogonal input + output dimension
       dropped

    Examples
    --------
    Typical use is in getting a 3D coordinate map from 4D

    >>> cm4d = AffineTransform.from_params('ijkl', 'xyzt', np.diag([1,2,3,4,1]))
    >>> cm3d = drop_io_dim(cm4d, 't')
    >>> cm3d.affine
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  2.,  0.,  0.],
           [ 0.,  0.,  3.,  0.],
           [ 0.,  0.,  0.,  1.]])
    '''
    aff = cm.affine.copy()
    in_dims = list(cm.function_domain.coord_names)
    out_dims = list(cm.function_range.coord_names)
    try:
        in_dim = in_dims.index(name)
    except ValueError:
        try:
            out_dim = out_dims.index(name)
        except ValueError:
            raise ValueError('No input or output dimension '
                             'with name (%s)' % name)
        else: # found out dimension, get in dimension
            in_dim, msg  = _matching_orth_dim(out_dim, aff)
    else: # found in dimension, get out dimension
        out_dim, msg = _matching_orth_dim(in_dim, aff.T)
    if None in (in_dim, out_dim):
        raise ValueError(msg)
    in_dims.pop(in_dim)
    out_dims.pop(out_dim)
    M, N = aff.shape
    rows = range(M)
    cols = range(N)
    rows.pop(out_dim)
    cols.pop(in_dim)
    aff = aff[rows]
    aff = aff[:,cols]
    return AffineTransform.from_params(in_dims, out_dims, aff)


def _matching_orth_dim(out_i, aff):
    ''' Find matching more or less orthogonal direction in affine

    If the corresponding affine row is all 0, and it is the only row
    that is all 0, then assume this is a 0 scaling rather than a dropped
    input coordinate, and work out which input coordinate (if any)
    remains to be claimed.

    Parameters
    ----------
    out_i : int
       Output (range) dimension
    aff : (M, N) array
       homogenous affine

    Returns
    -------
    in_i : None or int
       Matching orthogonal input (domain) dimension.  None if none
       matching.
    msg : str
       If `in_i` is None, `msg` returns informative message as to reason
       for failure to find matching index.

    Examples
    --------
    >>> aff = np.diag([1, 2, 3, 4, 1])
    >>> _matching_orth_dim(1, aff)
    (1, '')
    >>> aff = np.array([[0, 0, 0, 1, 10],
    ...                 [0, 0, 2, 0, 11],
    ...                 [0, 3, 0, 0, 12],
    ...                 [4, 0, 0, 0, 13],
    ...                 [0, 0, 0, 0, 1]])
    >>> _matching_orth_dim(1, aff)
    (2, '')
    '''
    rzs = aff[:-1,:-1]
    M, N = rzs.shape
    if out_i >= M:
        raise ValueError('%d row number is too high' % out_i)
    out_row = rzs[out_i]
    if np.allclose(out_row, 0):
        # zeros - is this zero scaling?
        msg = 'zeros in corresponding affine entries'
        # check for other zero rows - return failure if so
        out_rows = range(M)
        out_rows.pop(out_i)
        for i in out_rows:
            if np.allclose(rzs[i], 0):
                # another near zero row
                return None, msg + ' and other zero rows/cols'
        # what rows have been claimed already?
        ornt = io_orientation(aff[out_rows + [M]].T)
        candidates = set(range(N)) - set(ornt[:,0])
        if len(candidates) == 1:
            return candidates.pop(), ''
        return None, ' and corresponding index is ambiguous'
    in_i = np.argmax(np.abs(out_row))
    sel = range(N)
    sel.pop(in_i)
    if not np.allclose(out_row[sel], 0):
        return None, 'no corresponding orthogonal dimension'
    return in_i, ''


def append_io_dim(cm, in_name, out_name, start=0, step=1):
    ''' Append input and output dimension to coordmap

    Parameters
    ----------
    cm : Affine
       Affine coordinate map instance to which to append dimension
    in_name : str
       Name for new input dimension
    out_name : str
       Name for new output dimension
    start : float, optional
       Offset for transformed values in new dimension
    step : float, optional
       Step, or scale factor for transformed values in new dimension

    Returns
    -------
    cm_plus : Affine
       New coordinate map with appended dimension

    Examples
    --------
    Typical use is creating a 4D coordinate map from a 3D

    >>> cm3d = AffineTransform.from_params('ijk', 'xyz', np.diag([1,2,3,1]))
    >>> cm4d = append_io_dim(cm3d, 'l', 't', 9, 5)
    >>> cm4d.affine
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  0.,  0.,  0.],
           [ 0.,  0.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  5.,  9.],
           [ 0.,  0.,  0.,  0.,  1.]])
    '''
    # delayed import to avoid circular import errors
    from ...algorithms.utils.affines import append_diag
    aff = cm.affine
    in_dims = list(cm.function_domain.coord_names)
    out_dims = list(cm.function_range.coord_names)
    in_dims.append(in_name)
    out_dims.append(out_name)
    aff_plus = append_diag(aff, [step], [start])
    return AffineTransform.from_params(in_dims, out_dims, aff_plus)


class CoordMapMakerError(Exception):
    pass


class CoordMapMaker(object):
    """ Class to create coordinate maps of different dimensions
    """
    generic_maker = CoordinateMap
    affine_maker = AffineTransform

    def __init__(self, domain_maker, range_maker):
        """ Create coordinate map maker

        Parameters
        ----------
        domain_maker : callable
            A coordinate system maker, returning a coordinate system with input
            argument only ``N``, an integer giving the length of the coordinate
            map.
        range_maker : callable
            A coordinate system maker, returning a coordinate system with input
            argument only ``N``, an integer giving the length of the coordinate
            map.

        Examples
        --------
        >>> from nipy.core.reference.coordinate_system import CoordSysMaker
        >>> dmaker = CoordSysMaker('ijkl', 'generic-array')
        >>> rmaker = CoordSysMaker('xyzt', 'generic-scanner')
        >>> cm_maker = CoordMapMaker(dmaker, rmaker)
        """
        self.domain_maker = domain_maker
        self.range_maker = range_maker

    def make_affine(self, affine, append_zooms=(), append_offsets=()):
        """ Create affine coordinate map

        Parameters
        ----------
        affine : (M, N) array-like
            Array expressing the affine tranformation
        append_zooms : scalar or sequence length E
            If scalar, converted to sequence length E==1. Append E entries to
            the diagonal of `affine` (see examples)
        append_offsets : scalar or sequence length F
            If scalar, converted to sequence length F==1. If F==0, and E!=0, use
            sequence of zeros length E.  Append E entries to the translations
            (final column) of `affine` (see examples).

        Returns
        -------
        affmap : ``AffineTransform`` coordinate map

        Examples
        --------
        >>> from nipy.core.reference.coordinate_system import CoordSysMaker
        >>> dmaker = CoordSysMaker('ijkl', 'generic-array')
        >>> rmaker = CoordSysMaker('xyzt', 'generic-scanner')
        >>> cm_maker = CoordMapMaker(dmaker, rmaker)
        >>> cm_maker.make_affine(np.diag([2,3,4,1]))
        AffineTransform(
           function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='generic-array', coord_dtype=float64),
           function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='generic-scanner', coord_dtype=float64),
           affine=array([[ 2.,  0.,  0.,  0.],
                         [ 0.,  3.,  0.,  0.],
                         [ 0.,  0.,  4.,  0.],
                         [ 0.,  0.,  0.,  1.]])
        )

        We can add extra orthogonal dimensions, by specifying the diagonal
        elements:

        >>> cm_maker.make_affine(np.diag([2,3,4,1]), 6)
        AffineTransform(
           function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='generic-array', coord_dtype=float64),
           function_range=CoordinateSystem(coord_names=('x', 'y', 'z', 't'), name='generic-scanner', coord_dtype=float64),
           affine=array([[ 2.,  0.,  0.,  0.,  0.],
                         [ 0.,  3.,  0.,  0.,  0.],
                         [ 0.,  0.,  4.,  0.,  0.],
                         [ 0.,  0.,  0.,  6.,  0.],
                         [ 0.,  0.,  0.,  0.,  1.]])
        )

        Or the diagonal elements and the offset elements:

        >>> cm_maker.make_affine(np.diag([2,3,4,1]), [6], [9])
        AffineTransform(
           function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='generic-array', coord_dtype=float64),
           function_range=CoordinateSystem(coord_names=('x', 'y', 'z', 't'), name='generic-scanner', coord_dtype=float64),
           affine=array([[ 2.,  0.,  0.,  0.,  0.],
                         [ 0.,  3.,  0.,  0.,  0.],
                         [ 0.,  0.,  4.,  0.,  0.],
                         [ 0.,  0.,  0.,  6.,  9.],
                         [ 0.,  0.,  0.,  0.,  1.]])
        )
        """
        # delayed import to avoid circular import errors
        from ...algorithms.utils.affines import append_diag
        affine = np.asarray(affine)
        append_zooms = np.atleast_1d(append_zooms)
        append_offsets = np.atleast_1d(append_offsets)
        extra_N = len(append_zooms)
        if len(append_offsets) == 0:
            append_offsets = np.zeros(extra_N, dtype=append_zooms.dtype)
        elif len(append_offsets) != extra_N:
            raise CoordMapMakerError('Need same number of offsets as zooms')
        if extra_N != 0:
            affine = append_diag(affine, append_zooms, append_offsets)
        return self.affine_maker(self.domain_maker(affine.shape[1] - 1),
                                 self.range_maker(affine.shape[0] -1),
                                 affine)

    def make_cmap(self, domain_N, xform, inv_xform=None):
        """ Coordinate map with transform function `xform`

        Parameters
        ----------
        domain_N : int
            Number of domain coordinates
        xform : callable
            Function that transforms points of dimension `domain_N`
        inv_xform : None or callable, optional
            Function, such that ``inv_xform(xform(pts))`` returns ``pts``

        Returns
        -------
        cmap : ``CoordinateMap``

        Examples
        --------
        >>> from nipy.core.reference.coordinate_system import CoordSysMaker
        >>> dmaker = CoordSysMaker('ijkl', 'generic-array')
        >>> rmaker = CoordSysMaker('xyzt', 'generic-scanner')
        >>> cm_maker = CoordMapMaker(dmaker, rmaker)
        >>> cm_maker.make_cmap(4, lambda x : x+1) #doctest: +ELLIPSIS
        CoordinateMap(
           function_domain=CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='generic-array', coord_dtype=float64),
           function_range=CoordinateSystem(coord_names=('x', 'y', 'z', 't'), name='generic-scanner', coord_dtype=float64),
           function=<function <lambda> at ...>
          )
        """
        domain_cs = self.domain_maker(domain_N)
        ex_pt = np.zeros((1, domain_N), dtype=domain_cs.coord_dtype)
        xformed_pt = xform(ex_pt)
        range_N = xformed_pt.shape[1]
        return self.generic_maker(domain_cs,
                                  self.range_maker(range_N),
                                  xform,
                                  inv_xform)

    def __call__(self, *args, **kwargs):
        """ Create affine or non-affine coordinate map

        Parameters
        ----------
        \\*args :
            Arguments to ``make_affine`` or ``make_cmap`` methods. We check the
            first argument to see if it is a scalar or an affine, and pass the
            \\*args, \\*\\*kwargs to ``make_cmap`` or ``make_affine``
            respectively
        \\*\\*kwargs:
            See above

        Returns
        -------
        cmap : ``CoordinateMap`` or ``AffineTransform``
            Affine if the first \\*arg was an affine array, otherwise a
            Coordinate Map.
        """
        arg0 = np.asarray(args[0])
        if arg0.shape == ():
            return self.make_cmap(*args, **kwargs)
        return self.make_affine(*args, **kwargs)
