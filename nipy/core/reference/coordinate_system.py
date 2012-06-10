# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

class CoordinateSystemError(Exception):
    pass


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

    Two CoordinateSystems are equal if they have the same dtype
    and the same names and the same name.

    >>> another_cs = CoordinateSystem(names, 'not irrelevant', np.float)
    >>> cs == another_cs
    False
    >>> cs.dtype == another_cs.dtype
    True
    >>> cs.name == another_cs.name
    False
    """

    _doc = {}

    name = 'world-LPI'
    _doc['name'] = 'Name describing the CoordinateSystem'

    coord_names = ('x', 'y', 'z')
    _doc['coord_names'] = 'Tuple of names describing each coordinate.'

    coord_dtype = np.float64
    _doc['coord_dtype'] = 'The builtin, scalar,  dtype of each coordinate.'

    ndim = 3
    _doc['ndim'] = 'The number of dimensions'

    dtype = np.dtype([('x', np.float),
                      ('y', np.float),
                      ('z', np.float)])
    _doc['dtype'] = 'The composite dtype of the CoordinateSystem, ' + \
                    'expresses the fact that there are three numbers, the' + \
                    'first one corresponds to "x" and the second to "y".'

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
        CoordinateSystem(coord_names=('i', 'j'), name='input', coord_dtype=float64)
        >>> c.coord_dtype
        dtype('float64')
        """
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
        # Set all the attributes
        self.name = name
        self.coord_names = coord_names
        self.coord_dtype = coord_dtype
        self.ndim = len(coord_names)
        self.dtype = np.dtype([(name, self.coord_dtype) 
                               for name in self.coord_names])

    # All attributes are read only

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise AttributeError('the value of %s has already been set and all attributes are read-only' % key)
        object.__setattr__(self, key, value)

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
        """Equality is defined by self.dtype and self.name

        Parameters
        ----------
        other : :class:`CoordinateSystem`
           The object to be compared with

        Returns
        -------
        tf: bool
        """
        return (self.dtype == other.dtype) and (self.name == other.name)

    def similar_to(self, other):
        """Similarity is defined by self.dtype, ignoring name

        Parameters
        ----------
        other : :class:`CoordinateSystem`
           The object to be compared with

        Returns
        -------
        tf: bool
        """
        return (self.dtype == other.dtype)

    def __repr__(self):
        """Create a string representation of the coordinate system

        Returns
        -------
        s : string
        """
        return ("CoordinateSystem(coord_names=%s, name='%s', coord_dtype=%s)" %
                (self.coord_names, self.name, self.coord_dtype))


    def _checked_values(self, arr):
        ''' Check ``arr`` for valid dtype and shape as coordinate values.

        Raise Errors for failed checks.

        The dtype of ``arr`` has to be castable (without loss of precision) to
        ``self.coord_dtype``.  We use numpy ``can_cast`` for this check.

        The last (or only) axis of ``arr`` should be of length ``self.ndim``.

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
        >>> cs._checked_values(arr) # 1D is OK with matching dimensions 
        array([[1, 2, 3]], dtype=int16)
        >>> cs._checked_values(arr.reshape(1,3)) # as is 1 by N
        array([[1, 2, 3]], dtype=int16)
        >>> cs._checked_values(arr.reshape(3,1)) # wrong shape
        Traceback (most recent call last):
           ...
        CoordinateSystemError: Array shape[-1] (1) must match CoordinateSystem ndim (3).
          CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float32)
        >>> cs._checked_values(arr[0:2]) # wrong length
        Traceback (most recent call last):
           ...
        CoordinateSystemError: Array shape[-1] (2) must match CoordinateSystem ndim (3).
          CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float32)

        The dtype has to be castable:

        >>> cs._checked_values(np.array([1, 2, 3], dtype=np.float64))
        Traceback (most recent call last):
           ...
        CoordinateSystemError: Cannot cast array dtype float64 to CoordinateSystem coord_dtype float32.
          CoordinateSystem(coord_names=('i', 'j', 'k'), name='', coord_dtype=float32)

        The input array is unchanged, even if a reshape has
        occurred. The returned array points to the same data.

        >>> checked = cs._checked_values(arr)
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
        mistake (you were expecting an N-dimensional coordinate system), or it
        could be N points in 1D.  Because it is ambiguous, this is an error.

        >>> cs = CoordinateSystem('x')
        >>> cs._checked_values(1)
        array([[1]])
        >>> cs._checked_values([1, 2])
        Traceback (most recent call last):
           ...
        CoordinateSystemError: Array shape[-1] (2) must match CoordinateSystem ndim (1).
          CoordinateSystem(coord_names=('x',), name='', coord_dtype=float64)

        But of course 2D, N by 1 is OK

        >>> cs._checked_values(np.array([1,2,3]).reshape(3, 1))
        array([[1],
               [2],
               [3]])
        '''
        arr = np.atleast_2d(arr)
        if arr.shape[-1] != self.ndim:
            raise CoordinateSystemError('Array shape[-1] (%s) must match '
                                        'CoordinateSystem ndim (%d).\n  %s'
                                        % (arr.shape[-1], self.ndim, str(self)))
        if not np.can_cast(arr.dtype, self.coord_dtype):
            raise CoordinateSystemError('Cannot cast array dtype %s to '
                                        'CoordinateSystem coord_dtype %s.\n  %s' %
                                        (arr.dtype, self.coord_dtype, str(self)))
        return arr.reshape((-1, self.ndim))


def is_coordsys(obj):
    """ Test if `obj` has the CoordinateSystem API

    Parameters
    ----------
    obj : object
        Object to test

    Returns
    -------
    tf : bool
        True if `obj` has the coordinate system API

    Examples
    --------
    >>> is_coordsys(CoordinateSystem('xyz'))
    True
    >>> is_coordsys(CoordSysMaker('ikj'))
    False
    """
    if not hasattr(obj, 'coord_names'):
        return False
    if not hasattr(obj, 'name'):
        return False
    if not hasattr(obj, 'coord_dtype'):
        return False
    # Distinguish from CoordSysMaker
    return not callable(obj)


def safe_dtype(*dtypes):
    """Determine a dtype to safely cast all of the given dtypes to.

    Safe dtypes are valid numpy dtypes or python types which can be
    cast to numpy dtypes.  See numpy.sctypes for a list of valid
    dtypes.  Composite dtypes and string dtypes are not safe dtypes.

    Parameters
    ----------
    dtypes : sequence of ``np.dtype``

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
    TypeError: dtype must be valid numpy dtype bool, int, uint, float or complex

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
    TypeError: dtype must be valid numpy dtype bool, int, uint, float or complex
    """
    arrays = [np.zeros(2, dtype) for dtype in dtypes]
    kinds = [a.dtype.kind for a in arrays]
    if not set(kinds).issubset('iubfc'):
        raise TypeError('dtype must be valid numpy dtype bool, '
                        'int, uint, float or complex')
    return np.array(arrays).dtype


def product(*coord_systems, **kwargs):
    """Create the product of a sequence of CoordinateSystems.

    The coord_dtype of the result will be determined by ``safe_dtype``.

    Parameters
    ----------
    \*coord_systems : sequence of :class:`CoordinateSystem`
    name : str
        Name of ouptut coordinate system

    Returns
    -------
    product_coord_system : :class:`CoordinateSystem`

    Examples
    --------
    >>> c1 = CoordinateSystem('ij', 'input', coord_dtype=np.float32)
    >>> c2 = CoordinateSystem('kl', 'input', coord_dtype=np.complex)
    >>> c3 = CoordinateSystem('ik', 'in3')

    >>> print product(c1, c2)
    CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='product', coord_dtype=complex128)

    >>> print product(c1, c2, name='another name')
    CoordinateSystem(coord_names=('i', 'j', 'k', 'l'), name='another name', coord_dtype=complex128)

    >>> product(c2, c3)
    Traceback (most recent call last):
       ...
    ValueError: coord_names must have distinct names
    """
    name = kwargs.pop('name', 'product')
    if kwargs:
        raise TypeError('Unexpected kwargs %s' % kwargs)
    coords = []
    for c in coord_systems:
        coords += c.coord_names
    dtype = safe_dtype(*[c.coord_dtype for c in coord_systems])
    return CoordinateSystem(coords, name, coord_dtype=dtype)


class CoordSysMakerError(Exception):
    pass


class CoordSysMaker(object):
    """ Class to create similar coordinate maps of different dimensions
    """
    coord_sys_klass = CoordinateSystem

    def __init__(self, coord_names, name='', coord_dtype=np.float):
        """Create a coordsys maker with given axis `coord_names`

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
        >>> cmkr = CoordSysMaker('ijk', 'a name')
        >>> print cmkr(2)
        CoordinateSystem(coord_names=('i', 'j'), name='a name', coord_dtype=float64)
        >>> print cmkr(3)
        CoordinateSystem(coord_names=('i', 'j', 'k'), name='a name', coord_dtype=float64)
        """
        self.coord_names = tuple(coord_names)
        self.name = name
        self.coord_dtype = coord_dtype

    def __call__(self, N, name=None, coord_dtype=None):
        """ Create coordinate system of length `N`

        Parameters
        ----------
        N : int
            length of coordinate map
        name : None or str, optional
            Name of coordinate map.  Default is ``self.name``
        coord_dtype : None or dtype
            ``coord_dtype`` of returned coordinate system.  Default is
            ``self.coord_dtype``

        Returns
        -------
        csys : coordinate system
        """
        if name is None:
            name = self.name
        if coord_dtype is None:
            coord_dtype = self.coord_dtype
        if N > len(self.coord_names):
            raise CoordSysMakerError('Not enough axis names (have %d, '
                                     'you asked for %d)' %
                                     (len(self.coord_names), N))
        return self.coord_sys_klass(self.coord_names[:N], name, coord_dtype)


def is_coordsys_maker(obj):
    """ Test if `obj` has the CoordSysMaker API

    Parameters
    ----------
    obj : object
        Object to test

    Returns
    -------
    tf : bool
        True if `obj` has the coordinate system API

    Examples
    --------
    >>> is_coordsys_maker(CoordSysMaker('ikj'))
    True
    >>> is_coordsys_maker(CoordinateSystem('xyz'))
    False
    """
    if not hasattr(obj, 'coord_names'):
        return False
    if not hasattr(obj, 'name'):
        return False
    if not hasattr(obj, 'coord_dtype'):
        return False
    # Distinguish from CoordinateSystem
    return callable(obj)
