"""
Coordinate systems are used to represent the spaces in which the images reside.

A coordinate system contains coordinates.  For example a 3D coordinate system contains 3 coordinates: the first, second and third.

"""
__docformat__ = 'restructuredtext'

import copy, warnings
import numpy as np


class CoordinateSystem(object):
    """
    A CoordinateSystem is a (named) ordered sequence of coordinates,
    along with a dtype.


    """

    def __init__(self, coordinates, name= '', dtype=np.float):
        """
        Create a coordinate system with a given name and coordinate names.
        There are also two dtypes associated to the CoordinateSystem:
        one, self.value_dtype, which should be a numpy scalar dtype. The other,
        self.dtype, which is basically a description of the CoordinateSystem.

        :Parameters:
            coordinates : ``[Coordinate`]``
                The coordinates which make up the coordinate system
            name : ``string``
                The name of the coordinate system (optional)
            dtype : ``np.dtype``
                The dtype of the coordinates, should be a value_dtype scalar dtype.
        >>> c = CoordinateSystem('ij', name='input')
        >>> print c
        {'dtype': dtype('float64'), 'name': 'input', 'coordinates': ['i', 'j']}

        >>> c.value_dtype
        dtype('float64')
        >>> c.dtype
        dtype([('i', '<f8'), ('j', '<f8')])
        >>>                                        

        """
        self.name = name
        if len(set(coordinates)) != len(coordinates):
            raise ValueError, 'coordinates must have distinct names'

        # verify that the dtype is value_dtype for sanity
        sctypes = (np.sctypes['int'] + np.sctypes['float'] + 
                   np.sctypes['complex'] + np.sctypes['uint'])
        dtype = np.dtype(dtype)
        if dtype not in sctypes:
            raise ValueError, 'Coordinate dtype should be one of %s' % `sctypes`
        self._dtype = dtype
        self.coordinates = list(coordinates)

    def _getdtype(self):
        return np.dtype([(name, self._dtype) for name in self.coordinates])
    dtype = property(_getdtype, doc='dtype of CoordinateSystem with named fields')

    def _getvalue_dtype(self):
        return self._dtype
    value_dtype = property(
        _getvalue_dtype,
        doc='value_dtype scalar dtype of CoordinateSystem')

    def index(self, axisname):
        """
        Return the index of a given named coordinate.

        >>> c = CoordinateSystem('ij', name='input')
        >>> c.index('i')
        0

        """
        return self.coordinates.index(name)

    def rename(self, **kwargs):
        """
        Return a new CoordinateSystem with the coordinates renamed.

        >>> c = CoordinateSystem('ij', name='input')
        >>> print c
        {'dtype': dtype('float64'), 'name': 'input', 'coordinates': ['i', 'j']}
        >>> print c.rename(i='w')
        {'dtype': dtype('float64'), 'name': 'input-renamed', 'coordinates': ['w', 'j']}
        """
        coords = []
        for a in self.coordinates:
            if a in kwargs.keys():
                coords.append(kwargs[a])
            else:
                coords.append(a)


        if self.name:
            name = self.name + '-renamed'
        else:
            name = ''
        return CoordinateSystem(coords, name, self.value_dtype)

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
        _dict = {'name': self.name,
                 'coordinates':self.coordinates,
                 'dtype':self.value_dtype}
        return `_dict`
   
    def _getndim(self):
        """ Number of dimensions 
        
        :Returns: ``int``
        """
        return len(self.coordinates)
    ndim = property(_getndim)
    
    def typecast(self, x, dtype=None):
        """
        Try to safely typecast x into
        an ndarray with a numpy value_dtype dtype
        with the correct shape, or
        typecast it as an ndarray with self.dtype.

        """
        x = np.asarray(x)

        if dtype not in [self.dtype, self.value_dtype]:
            raise ValueError, 'only safe to cast to either %s or %s' % (`self.dtype`, `self.value_dtype`)
        if x.dtype not in [self.dtype, self.value_dtype]:
            raise ValueError, 'only safe to cast from either %s or %s' % (`self.dtype`, `self.value_dtype`)

        if dtype == self.dtype:
            if x.dtype == self.dtype: # do nothing
                return x
            
            # this presumes
            # we are given an ndarray
            # with dtype = self.value_dtype
            # so we typecast, to be safe we make a copy!

            x = np.asarray(x)
            shape = x.shape

            # The last shape entry should match the length
            # of self.dtype

            if x.shape[-1] != len(self.dtype.names):
                warnings.warn("dangerous typecast, shape is unexpected: %d, %d" % (x.shape[-1], len(self.dtype.names)))

            x = np.asarray(x, dtype=self.value_dtype).ravel()
            y = x.view(self.dtype)
            y.shape = shape[:-1]
            return y
        else:
            if x.dtype == self.value_dtype: # do nothing
                return x
            y = x.ravel().view(self.value_dtype)
            y.shape = x.shape + (y.shape[0] / np.product(x.shape),)
            return y

    def reorder(self, name=None, order=None):
        """
        Given a name for the reordered coordinates, and a new order, return a
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
        {'dtype': dtype('float64'), 'name': 'input', 'coordinates': ['k', 'i', 'j']}

        """
        if order is None:
            order = range(len(self.coordinates))[::-1]
        if name is None:
            name = self.name
        return CoordinateSystem(_reorder(self.coordinates, order),
                                name,
                                self.value_dtype)

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

    >>> c1 = CoordinateSystem('ij', 'input', dtype=np.float32)
    >>> c2 = CoordinateSystem('kl', 'input', dtype=np.complex)
    >>> safe_dtype(c1.value_dtype, c2.value_dtype)
    dtype('complex128')

    """
    arrays = [np.zeros(2, dtype) for dtype in dtypes]
    notbuiltin = filter(lambda x: not x.dtype.isbuiltin, arrays)
    if notbuiltin:
        raise ValueError('dtypes must be value_dtype')
    return np.array(arrays).dtype

def product(*coord_systems):
    """
    Create the product of a sequence of CoordinateSystems.
    The value_dtype dtype of the result will be determined by safe_dtype.

    >>> c1 = CoordinateSystem('ij', 'input', dtype=np.float32)
    >>> c2 = CoordinateSystem('kl', 'input', dtype=np.complex)
    >>> c3 = CoordinateSystem('ik', 'in3')

    >>> print product(c1,c2)
    {'dtype': dtype('complex128'), 'name': 'product', 'coordinates': ['i', 'j', 'k', 'l']}

    >>> try:
    ...     product(c2,c3)
    ... except ValueError, msg:
    ...     print 'Error: %s' % msg
    ...     pass
    ...
    Error: coordinates must have distinct names
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
        coords += c.coordinates
    dtype = safe_dtype(*[c.value_dtype for c in coord_systems])
    return CoordinateSystem(coords, 'product', dtype=dtype)
