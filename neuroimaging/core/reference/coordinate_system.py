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
    A CoordinateSystem is a (named) ordered sequence of coordinates,
    along with a dtype.


    """

    def __init__(self, coord_names, name= '', coord_dtype=np.float):
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
           The dtype of the coord_names.  This should be a built-in numpy
           scalar dtype. (default is np.float)

        Examples
        --------
        >>> c = CoordinateSystem('ij', name='input')
        >>> print c
        name: input, coord_names: ['i', 'j'], coord_dtype: float64
        
        >>> c.coord_dtype
        dtype('float64')
        >>> c.dtype
        dtype([('i', '<f8'), ('j', '<f8')])

        """
        self.name = name
        # this allows coord_names to be an iterator and have a length
        coord_names = list(coord_names)
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
        self.coord_names = coord_names

    def _getdtype(self):
        return np.dtype([(name, self._coord_dtype) 
                         for name in self.coord_names])
    dtype = property(_getdtype, 
                     doc='dtype of CoordinateSystem with named fields')

    def _getcoord_dtype(self):
        return self._coord_dtype
    coord_dtype = property(
        _getcoord_dtype,
        doc='coord_dtype scalar dtype of CoordinateSystem')

    def index(self, axisname):
        """
        Return the index of a given named coordinate.

        >>> c = CoordinateSystem('ij', name='input')
        >>> c.index('i')
        0

        """
        return self.coord_names.index(axisname)

    def rename(self, **kwargs):
        """
        Return a new CoordinateSystem with the coord_names renamed.

        >>> c = CoordinateSystem('ij', name='input')
        >>> print c
        name: input, coord_names: ['i', 'j'], coord_dtype: float64
        
        >>> print c.rename(i='w')
        name: input-renamed, coord_names: ['w', 'j'], coord_dtype: float64

        """

        coords = []
        for a in self.coord_names:
            if a in kwargs.keys():
                coords.append(kwargs[a])
            else:
                coords.append(a)


        if self.name:
            name = self.name + '-renamed'
        else:
            name = ''
        return CoordinateSystem(coords, name, self.coord_dtype)

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
        repr : string
        """
        
        attrs = ['name', 'coord_names', 'coord_dtype']
        vals = []
        for attr in attrs:
            vals.append('%s: %s' % (attr, getattr(self, attr)))
        return ', '.join(vals)

    def _getndim(self):
        """ Number of dimensions 
        
        :Returns: ``int``
        """
        return len(self.coord_names)
    ndim = property(_getndim)
    
    def typecast(self, x, dtype=None):
        """
        Try to safely typecast x into
        an ndarray with a numpy coord_dtype dtype
        with the correct shape, or
        typecast it as an ndarray with self.dtype.

        """
        x = np.asarray(x)

        if dtype not in [self.dtype, self.coord_dtype]:
            raise ValueError, 'only safe to cast to either %s or %s' % (`self.dtype`, `self.coord_dtype`)
        if x.dtype not in [self.dtype, self.coord_dtype]:
            raise ValueError, 'only safe to cast from either %s or %s' % (`self.dtype`, `self.coord_dtype`)

        if dtype == self.dtype:
            if x.dtype == self.dtype: # do nothing
                return x
            
            # this presumes
            # we are given an ndarray
            # with dtype = self.coord_dtype
            # so we typecast, to be safe we make a copy!

            x = np.asarray(x)
            shape = x.shape

            # The last shape entry should match the length
            # of self.dtype

            if x.shape[-1] != len(self.dtype.names):
                warnings.warn("dangerous typecast, shape is unexpected: %d, %d" % (x.shape[-1], len(self.dtype.names)))

            x = np.asarray(x, dtype=self.coord_dtype).ravel()
            y = x.view(self.dtype)
            y.shape = shape[:-1]
            return y
        else:
            if x.dtype == self.coord_dtype: # do nothing
                return x
            y = x.ravel().view(self.coord_dtype)
            y.shape = x.shape + (y.shape[0] / np.product(x.shape),)
            return y

    def reorder(self, name=None, order=None):
        """
        Given a name for the reordered coord_names, and a new order, return a
        reordered coordinate system. Defaults to reversal.

        Parameters
        ----------
        name : string
           The name for the new coordinate system
        order : sequence of int
           The order of the axes, e.g. [2, 0, 1]

        Returns
        -------
        reordered : CoordinateSystem

        Examples
        --------
        >>> c = CoordinateSystem('ijk', name='input')
        >>> print c.reorder(order=[2,0,1])
        name: input, coord_names: ['k', 'i', 'j'], coord_dtype: float64
        
        """
        if order is None:
            order = range(len(self.coord_names))[::-1]
        if name is None:
            name = self.name
        return CoordinateSystem(_reorder(self.coord_names, order),
                                name,
                                self.coord_dtype)

def _reorder(seq, order):
    """ Reorder a sequence. """
    return [seq[i] for i in order]

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
    The coord_dtype dtype of the result will be determined by safe_dtype.

    >>> c1 = CoordinateSystem('ij', 'input', coord_dtype=np.float32)
    >>> c2 = CoordinateSystem('kl', 'input', coord_dtype=np.complex)
    >>> c3 = CoordinateSystem('ik', 'in3')

    >>> print product(c1,c2)
    name: product, coord_names: ['i', 'j', 'k', 'l'], coord_dtype: complex128

    >>> try:
    ...     product(c2,c3)
    ... except ValueError, msg:
    ...     print 'Error: %s' % msg
    ...     pass
    ...
    Error: coord_names must have distinct names
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
