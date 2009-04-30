"""
CoordinateSystems are used to represent the space in which the image resides.

A CoordinateSystem contains named coordinates, one for each dimension
and a coordinate dtype.  The purpose of the CoordinateSystem is to
specify the name and order of the coordinate axes for a particular
space.  This allows one to compare two CoordinateSystems to determine
if they are equal.

"""
__docformat__ = 'restructuredtext'

import numpy as np


class CoordinateSystem(object):
    """An ordered sequence of named coordinates of a specified dtype.

    A coordinate system is defined by the names of the coordinates,
    (attribute ``coord_names``) and the numpy dtype of each coordinate
    value (attribute ``coord_dtype``).  The coordinate system can also
    have a name.

    >>> names = ['first', 'second', 'third']
    >>> cs = CoordinateSystem(names, 'a coordinate system', np.float)
    >>> cs.coord_names
    ('first', 'second', 'third')
    >>> cs.name
    'a coordinate system'
    >>> cs.coord_dtype
    dtype('float64')

    The coordinate system also has a ``dtype`` which is the composite
    numpy dtype, made from the (``names``, ``coord_dtype``).

    >>> dtype_template = [(name, np.float) for name in cs.coord_names]
    >>> dtype_should_be = np.dtype(dtype_template)
    >>> cs.dtype == dtype_should_be
    True

    Two CoordinateSystems are equal if they have the same dtype.  The
    CoordinateSystem names may be different.

    >>> another_cs = CoordinateSystem(names, 'irrelevant', np.float)
    >>> cs == another_cs
    True
    >>> cs.dtype == another_cs.dtype
    True
    >>> cs.name == another_cs.name
    False

    """

    def __init__(self, coord_names, name='', coord_dtype=np.float):
        """Create a coordinate system with a given name and coordinate names.
        
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
            raise ValueError('coord_names must have distinct names')

        # verify that the dtype is coord_dtype for sanity
        sctypes = (np.sctypes['int'] + np.sctypes['float'] + 
                   np.sctypes['complex'] + np.sctypes['uint'])
        coord_dtype = np.dtype(coord_dtype)
        if coord_dtype not in sctypes:
            raise ValueError('Coordinate dtype should be one of %s' % `sctypes`)
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
        """Equality is defined by self.dtype.

        Parameters
        ----------
        other : :class:`CoordinateSystem`
           The object to be compared with

        Returns
        -------
        tf: bool

        """

        return (self.dtype == other.dtype)

    def __str__(self):
        """Create a string representation of the coordinate system

        Returns
        -------
        s : string

        """
        
        attrs = ('name', 'coord_names', 'coord_dtype')
        vals = []
        return ("name: '%s', coord_names: %s, coord_dtype: %s" %
                (self.name, self.coord_names, self.coord_dtype))


    def _checked_values(self, arr):
        ''' Check ``arr`` for valid dtype and shape as coordinate values.

        Raise Errors for failed checks.

        The dtype of ``arr`` has to be castable (without loss of
        precision) to ``self.coord_dtype``.  We use numpy ``can_cast``
        for this check.

	The last (or only) axis of ``arr`` should be of length
	``self.ndim``.

	Parameters
	----------
	arr : array-like
	   array to check
	
	Returns
	-------
	checked_arr : array
           Possibly reshaped array

	Examples
	--------
	>>> cs = CoordinateSystem('ijk', coord_dtype=np.float32)
        >>> arr = np.array([1, 2, 3], dtype=np.int16)
        >>> cs.checked_values(arr.reshape(1,3)) 
        array([[1, 2, 3]], dtype=int16)
        >>> cs.checked_values(arr) # 1D is OK with matching dimensions 
        array([[1, 2, 3]], dtype=int16)
        >>> cs.checked_values(arr.reshape(3,1)) # wrong shape
        Traceback (most recent call last):
           ...
        ValueError: Array shape[-1] must match CoordinateSystem shape 3.
          name: '', coord_names: ('i', 'j', 'k'), coord_dtype: float32

        >>> cs.checked_values(arr[0:2]) # wrong length
        Traceback (most recent call last):
           ...
        ValueError: 1D input should have length 3 for CoordinateSystem:
          name: '', coord_names: ('i', 'j', 'k'), coord_dtype: float32

        The dtype has to be castable:

        >>> cs.checked_values(np.array([1, 2, 3], dtype=np.float64))
        Traceback (most recent call last):
           ...
        ValueError: Cannot cast array dtype float64 to CoordinateSystem coord_dtype float32.
          name: '', coord_names: ('i', 'j', 'k'), coord_dtype: float32

        The input array is unchanged, even if a reshape has
        occurred. The returned array points to the same data.

        >>> checked = cs.checked_values(arr)
        >>> checked.shape == arr.shape
        False
        >>> checked is arr
        False
        >>> arr[0]
        1
        >>> checked[0,0] = 10
        >>> arr[0]
        10

        For a 1D CoordinateSystem, passing a 1D vector length N could be a
        mistake (you were expecting an N-dimensional coordinate
        system), or it could be N points in 1D.  Because it is
        ambiguous, this is an error.

        >>> cs = CoordinateSystem('x')
        >>> cs.checked_values(1)
        array([[1]])
        >>> cs.checked_values([1, 2])
        Traceback (most recent call last):
           ...
        ValueError: 1D input should have length 1 for CoordinateSystem:
          name: '', coord_names: ('x',), coord_dtype: float64

        But of course 2D, N by 1 is OK

        >>> cs.checked_values(np.array([1,2,3]).reshape(3, 1))
        array([[1],
               [2],
               [3]])
        '''
        arr = np.asanyarray(arr)
        our_ndim = len(self._coord_names)
        if len(arr.shape) < 2:
            if arr.size != our_ndim:
                raise ValueError('1D input should have length %d for '
                                 'CoordinateSystem:\n  %s' % 
                                 (our_ndim, str(self)))
            arr = arr.reshape((1, arr.size))
        elif arr.shape[-1] != our_ndim:
            raise ValueError('Array shape[-1] must match CoordinateSystem '
                             'shape %d.\n  %s' % (our_ndim, str(self)))
        if not np.can_cast(arr.dtype, self._coord_dtype):
            raise ValueError('Cannot cast array dtype %s to '
                             'CoordinateSystem coord_dtype %s.\n  %s' %
                             (arr.dtype, self._coord_dtype, str(self)))
        return arr


def safe_dtype(*dtypes):
    """Determine a dtype to safely cast all of the given dtypes to.

    Safe dtypes are valid numpy dtypes or python types which can be
    cast to numpy dtypes.  See numpy.sctypes for a list of valid
    dtypes.  Composite dtypes and string dtypes are not safe dtypes.

    To see if your dtypes are valid, build a numpy array of each dtype
    and the resulting object should return *1* from the
    varname.dtype.isbuiltin attribute.

    Parameters
    ----------
    dtypes : sequence of builtin ``np.dtype``

    Returns
    -------
    dtype : np.dtype

    Examples
    --------

    >>> c1 = CoordinateSystem('ij', 'input', coord_dtype=np.float32)
    >>> c2 = CoordinateSystem('kl', 'input', coord_dtype=np.complex)
    >>> safe_dtype(c1.coord_dtype, c2.coord_dtype)
    dtype('complex128')

    >>> # Strings are invalid dtypes
    >>> safe_dtype(type('foo'))
    Traceback (most recent call last):
    ...
    TypeError: dtype must be valid numpy dtype int, uint, float or complex

    >>> # Check for a valid dtype
    >>> myarr = np.zeros(2, np.float32)
    >>> myarr.dtype.isbuiltin
    1

    >>> # Composite dtypes are invalid
    >>> mydtype = np.dtype([('name', 'S32'), ('age', 'i4')])
    >>> myarr = np.zeros(2, mydtype)
    >>> myarr.dtype.isbuiltin
    0
    >>> safe_dtype(mydtype)
    Traceback (most recent call last):
    ...
    TypeError: dtype must be valid numpy dtype int, uint, float or complex

    """

    arrays = [np.zeros(2, dtype) for dtype in dtypes]
    notbuiltin = filter(lambda x: not x.dtype.isbuiltin, arrays)
    if notbuiltin:
        raise TypeError('dtype must be valid numpy dtype int, uint, float or complex')
    return np.array(arrays).dtype


def product(*coord_systems):
    """Create the product of a sequence of CoordinateSystems.

    The coord_dtype of the result will be determined by ``safe_dtype``.

    Parameters
    ----------
    coord_systems : sequence of :class:`CoordinateSystem`
    
    Returns
    -------
    product_coord_system : :class:`CoordinateSystem`

    Examples
    --------

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

    """

    coords = []
    for c in coord_systems:
        coords += c.coord_names
    dtype = safe_dtype(*[c.coord_dtype for c in coord_systems])
    return CoordinateSystem(coords, 'product', coord_dtype=dtype)
