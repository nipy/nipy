"""
Coordinate systems are used to represent the spaces in which the images reside.

A coordinate system contains coordinates.  For example a 3D coordinate
system contains 3 coordinates: the first, second and third.

"""
__docformat__ = 'restructuredtext'

import copy, warnings
import numpy as np


class CoordinateSystem(object):
    """
    A CoordinateSystem is an ordered sequence of named coordinates,
    along with a numpy dtype for the coordinate values.

    A coordinate system is defined by the names of the coordinates,
    (attribute ``coord_name``) and the coordinate numpy dtype
    (attribute ``coord_dtype``).

    It can also have a name.

    >>> names = ['first', 'second', 'third']
    >>> cs = CoordinateSystem(names, 'a coordinate system', np.float)
    >>> cs.coord_names
    ('first', 'second', 'third')
    >>> cs.name
    'a coordinate system'
    >>> cs.coord_dtype
    dtype('float64')

    The attribute ``dtype`` gives the composite numpy type, made from
    the (``names``, ``coord_dtype``).

    >>> dtype_template = [(name, np.float) for name in cs.coord_names]
    >>> dtype_should_be = np.dtype(dtype_template)
    >>> cs.dtype == dtype_should_be
    True

    By "defines the coordinate system", we mean, defines equality.

    >>> another_cs = CoordinateSystem(names, 'irrelevant', np.float)
    >>> cs == another_cs
    True
    >>> cs.dtype == another_cs.dtype
    True
    >>> cs.name == another_cs.name
    False
    """

    def __init__(self, coord_names, name='', coord_dtype=np.float):
        """
        Create a coordinate system with a given name and coordinate names.
        The CoordinateSystem has two dtype attributes:
        
        #. self.coord_dtype is the dtype of the individual coordinate values
        #. self.dtype is the recarray dtype for the CoordinateSystem
           which combines the coord_names and the coord_dtype.  This
           functions as the description of the CoordinateSystem.

        Parameters
        ----------
        coord_names : iterable
           A sequence of coordinate names.
        name : string, optional
           The name of the coordinate system
        coord_dtype : np.dtype, optional
           The dtype of the coord_names.  This should be a built-in
           numpy scalar dtype. (default is np.float).  The value can
           by anything that can be passed to the np.dtype constructor.
           For example ``np.float``, ``np.dtype(np.float)`` or ``f8``
           all result in the same ``coord_dtype``.

        Examples
        --------
        >>> c = CoordinateSystem('ij', name='input')
        >>> print c
        name: 'input', coord_names: ('i', 'j'), coord_dtype: float64
        
        >>> c.coord_dtype
        dtype('float64')

        """
        self.name = name
        # this allows coord_names to be an iterator and have a length
        coord_names = tuple(coord_names)
        # Make sure each coordinate is unique
        if len(set(coord_names)) != len(coord_names):
            raise ValueError, 'coord_names must have distinct names'

        # verify that the dtype is coord_dtype for sanity
        sctypes = (np.sctypes['int'] + np.sctypes['float'] + 
                   np.sctypes['complex'] + np.sctypes['uint'])
        coord_dtype = np.dtype(coord_dtype)
        if coord_dtype not in sctypes:
            raise ValueError, 'Coordinate dtype should be one of %s' % `sctypes`
        self._coord_dtype = coord_dtype
        self._coord_names = coord_names

    def _get_dtype(self):
        return np.dtype([(name, self._coord_dtype) 
                         for name in self.coord_names])
    dtype = property(_get_dtype, 
                     doc='The dtype of the CoordinateSystem.')

    def _get_coord_dtype(self):
        return self._coord_dtype
    coord_dtype = property(_get_coord_dtype,
                           doc='The dtype of the coordinates in the CoordinateSytem')

    def _get_coord_names(self):
        return self._coord_names
    coord_names = property(_get_coord_names,
                           doc='The coordinate names in the CoordinateSystem')

    def _get_ndim(self):
        return len(self.coord_names)
    ndim = property(_get_ndim,
                    doc='The number of coordinates in the CoordinateSystem')
    
    def index(self, coord_name):
        """Return the index of a given named coordinate.

        >>> c = CoordinateSystem('ij', name='input')
        >>> c.index('i')
        0

        >>> c.index('j')
        1

        """
        return list(self.coord_names).index(coord_name)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        """
        Equality is defined by self.dtype.

        Parameters
        ----------
        other : CoordinateSystem
           The object to be compared with

        Returns
        -------
        tf: bool
        """
        return (self.dtype == other.dtype)

    def __str__(self):
        """
        Create a string representation of the coordinate system

        Returns
        -------
        s : string
        """
        
        attrs = ('name', 'coord_names', 'coord_dtype')
        vals = []
        return ("name: '%s', coord_names: %s, coord_dtype: %s" %
                (self.name, self.coord_names, self.coord_dtype))

    def typecast_values(self, x, dtype=None):
        """ Try to safely typecast array-like object x 

        Typecast ``x`` into an ndarray with numpy dtype
        ``coord_dtype``, and with the correct shape, or typecast it as
        an ndarray with the composite coordinate system dtype,
        ``self.dtype``.

        Parameters
        ----------
        x : array-like
           array of coordinate values
        dtype : np.dtype, optional
           Requested output dtype of array.  If no dtype is specified,
           use ``self.coord_dtype``.
        
        Returns
        -------
        tca : array
           numpy array with correct shape, and dtype

        Examples
        --------
        >>> cs = CoordinateSystem('ijk', coord_dtype=np.float)
        >>> arr = np.zeros((10,3),dtype=np.float)

        If the array is already the right shape and type, we pass it
        through unmodified.

        >>> tcarr = cs.typecast_values(arr)
        >>> np.all(tcarr == arr)
        True
        >>> tcarr is arr
        True

        If the cast array is of dtype ``coord_dtype``, then the last
        element of the shape of ``x`` should match the number of
        dimensions of the coordinate system.

        >>> arr = np.zeros((5,6),dtype=np.float)
        >>> tcarr = cs.typecast_values(arr)
        Traceback (most recent call last):
           ...
        ValueError: value arrays should be 2d with final dimension matching coordinate system dimension
        """
        x = np.asarray(x)
        if dtype is None:
            dtype = self.coord_dtype
        else:
            # we need a numpy dtype to do the comparison
            dtype = np.dtype(dtype)
            if dtype not in [self.dtype, self.coord_dtype]:
                raise ValueError('only safe to cast to either %s or %s' 
                                 % (`self.dtype`, `self.coord_dtype`))
        if x.dtype not in [self.dtype, self.coord_dtype]:
            raise ValueError('only safe to cast from either %s or %s'
                             % (`self.dtype`, `self.coord_dtype`))
        # Check input shape
        if x.dtype == self.coord_dtype: # coordinate dtype
            if len(x.shape) !=2 or x.shape[1] != self.ndim:
                raise ValueError('value arrays should be 2d '
                                 'with final dimension matching '
                                 'coordinate system dimension')
        else: # coordinate system dtype
            if len(x.shape) != 1:
                raise ValueError('structured arrays should be 1d')
        if dtype == self.dtype:
            # we want a composite type
            if x.dtype == self.dtype:
                # x is already a structured array (composite dtype)
                return x
            # we need to cast x to the composite dtype
            # this presumes we are given an ndarray with dtype =
            # self.coord_dtype so we typecast, to be safe we make a
            # copy!
            shape = x.shape
            x = np.asarray(x, dtype=self.coord_dtype).ravel()
            y = x.view(self.dtype)
            y.shape = shape[:-1]
            return y
        else: # casting to the builtin coord_dtype
            if x.dtype == dtype: 
                return x
            # we need to cast to coord_dtype
            y = x.ravel().view(self.coord_dtype)
            y.shape = x.shape + (y.shape[0] / np.product(x.shape),)
            return y


def safe_dtype(*dtypes):
    """
    Try to determine a dtype to which all of the dtypes can safely be
    typecast by creating an array with elements of all of these dtypes.

    Parameters
    ----------
    dtypes : sequence of builtin ``np.dtype``s

    Returns
    -------
    dtype: np.dtype

    >>> c1 = CoordinateSystem('ij', 'input', coord_dtype=np.float32)
    >>> c2 = CoordinateSystem('kl', 'input', coord_dtype=np.complex)
    >>> safe_dtype(c1.coord_dtype, c2.coord_dtype)
    dtype('complex128')

    """
    arrays = [np.zeros(2, dtype) for dtype in dtypes]
    notbuiltin = filter(lambda x: not x.dtype.isbuiltin, arrays)
    if notbuiltin:
        raise ValueError('dtypes must be coord_dtype')
    return np.array(arrays).dtype


def product(*coord_systems):
    """
    Create the product of a sequence of CoordinateSystems.

    The coord_dtype dtype of the result will be determined by
    ``safe_dtype``.

    >>> c1 = CoordinateSystem('ij', 'input', coord_dtype=np.float32)
    >>> c2 = CoordinateSystem('kl', 'input', coord_dtype=np.complex)
    >>> c3 = CoordinateSystem('ik', 'in3')

    >>> print product(c1,c2)
    name: 'product', coord_names: ('i', 'j', 'k', 'l'), coord_dtype: complex128

    >>> product(c2,c3)
    Traceback (most recent call last):
       ...
    ValueError: coord_names must have distinct names
    >>>                     


    :Inputs:
    --------
    coord_systems: sequence of ``CoordinateSystem``s
    
    :Returns:
    ---------
    product_coord_system: CoordinateSystem

    """
    coords = []
    for c in coord_systems:
        coords += c.coord_names
    dtype = safe_dtype(*[c.coord_dtype for c in coord_systems])
    return CoordinateSystem(coords, 'product', coord_dtype=dtype)
