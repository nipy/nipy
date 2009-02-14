"""
Coordinate systems are used to represent the spaces in which the images reside.

A coordinate system contains coordinates.  For example a 3D coordinate
system contains 3 coordinates: the first, second and third.

"""
__docformat__ = 'restructuredtext'

import copy, warnings
import numpy as np

class Coordinate(object):
    """
    This class represents a generic coordinate. 

    A coordinate has a name and a builtin dtype, i.e. ('x', np.float).
    ``Coordinate``s are used in the definition of ``CoordinateSystem``.
    """

    def __str__(self):
        return '<Coordinate:"%(name)s", dtype=%(dtype)s>' % \
            {'name':self.name, 'dtype':`self.dtype.descr`}

    def __repr__(self):
        return self.__str__()

    def __init__(self, name, dtype=np.float):
        """
        Create a Coordinate with a given name and dtype.

        Parameters
        ----------
        name : string
           The name for the coordinate.
        dtype : np.dtype
           The dtype of the axis. Must be a builtin dtype.
        """
        self.name = name
        self._dtype = np.dtype(dtype)

        # verify that the dtype is builtin for sanity
        if not self._dtype.isbuiltin:
            raise ValueError, 'Coordinate dtypes should be numpy builtin dtypes'
    def _getdtype(self):
        return np.dtype([(self.name, self._dtype)])
    def _setdtype(self, dtype):
        self._dtype = dtype
    dtype = property(_getdtype, _setdtype, doc='Named dtype of a Coordinate.')

    def _getbuiltin(self):
        return self._dtype
    builtin = property(_getbuiltin, doc='Numpy builtin dtype of a Coordinate.')

    def _getndim(self):
        """ Number of dimensions 
        
        Returns
        -------
        ndims : int
        """
        return len(self.axes)
    ndim = property(_getndim)
    
    def __eq__(self, other):
        """ Equality is defined by dtype.

        Parameters
        ----------
        other : Coordinate
           The object to be compared with.

        Returns
        -------
        tf : bool
        """
        return self.dtype == other.dtype


class CoordinateSystem(object):
    """
    A CoordinateSystem is a (named) ordered sequence of ``Coordinate``s.
    """

    def __init__(self, name, coordinates):
        """
        Create a coordinate system with a given name and axes.

	Parameters
	----------
	name : ``string``
           The name of the coordinate system
        coordinates : sequence of coordinates
           The coordinates which make up the coordinate system
        """
        self.name = name
        if len(set([ax.name for ax in coordinates])) != len(coordinates):
            raise ValueError, 'coordinates must have distinct names'
        dtype = safe_dtype(*tuple([ax.builtin for ax in coordinates]))
        values = []
        for ax in coordinates:
            ax = copy.copy(ax)
            ax.dtype = dtype
            values.append(ax)
        self.coordinates = tuple(values)

    def _getdtype(self):
        return np.dtype([(ax.name, ax.builtin) for ax in self.axes])
    dtype = property(_getdtype)

    def _getbuiltin(self):
        d = self.dtype.descr
        different = filter(lambda x: x[1] != d[0][1], d)
        if not different:
            d = np.dtype(d[0][1])
            if d.isbuiltin:
                return d
            else:
                raise ValueError(
                    'could not work out a builtin dtype for this coordinate system')
    builtin = property(_getbuiltin)

    def _getaxisnames(self):
        """ A list of the names of the coordinate system's axes. 
        
        Returns 
        -------
        names : sequence of strings
        """
        return [co.name for co in self.coordinates]
    axisnames = property(_getaxisnames)
        
    def _getaxes(self):
        """ A list of the coordinate system's axes. 
        
        Returns
        -------
        axes : sequence of coordinates (axes)
        """
        return self.coordinates
    axes = property(_getaxes)

    def __getitem__(self, axisname):
        """
        Return an axis indexed by name

        Parameters
        ----------
        axisname : string
           The name of the axis to return

        Returns
        -------
        coord : coordinate
           coordinate (axis) with given ``axisname``
        
        Notes
        -----
        Raises KeyError: If axisname is not the name of an axis in
            this coordinate system.
        """
        for co in self.coordinates:
            if axisname == co.name:
                return co
        raise KeyError(
            "axis '%s' not found, names are %s"%(axisname,self.axisnames))

    def index(self, axisname):
        """
        Return the index of a given axisname.
        """
        for i, co in enumerate(self.coordinates):
            if axisname == co.name:
                return i
        raise KeyError(
            "axis '%s' not found, names are %s" %
            (axisname, self.axisnames))

    def rename(self, **kwargs):
        """
        Return a new CoordinateSystem with the values renamed.

        >>> axes = [Coordinate(n) for n in 'abc']
        >>> coords = CoordinateSystem('input', axes)
        >>> print coords.rename(a='x')
        {'axes': (<Coordinate:"x", dtype=[('x', '<f8')]>, <Coordinate:"b", dtype=[('b', '<f8')]>, <Coordinate:"c", dtype=[('c', '<f8')]>), 'name': 'input-renamed'}
        >>>                                               
        """
        axes = []
        for a in self.axisnames:
            axis = copy.copy(self[a])
            if a in kwargs.keys():
                axis.name = kwargs[a]
            axes.append(axis)
        return CoordinateSystem(self.name + '-renamed', axes)

    def __setitem__(self, name, value):
        """
        Setting of index values is not allowed.
        
        :Raises TypeError: Always.
        """
        raise TypeError, "CoordinateSystem does not support axis assignment"

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
        return self.dtype == other.dtype

    def __repr__(self):
        """
        Create a string representation of the coordinate system

        Returns
        -------
        repr : string
        """
        
        _dict = {'name': self.name,
                 'axes': self.coordinates}
        return `_dict`
   
    def typecast(self, x, dtype=None):
        """
        Try to safely typecast x into
        an ndarray with a numpy builtin dtype
        with the correct shape, or
        typecast it as an ndarray with self.dtype.

        """
        x = np.asarray(x)

        if dtype not in [self.dtype, self.builtin]:
            raise ValueError, 'only safe to cast to either %s or %s' % (`self.dtype`, `self.builtin`)
        if x.dtype not in [self.dtype, self.builtin]:
            raise ValueError, 'only safe to cast from either %s or %s' % (`self.dtype`, `self.builtin`)

        if dtype == self.dtype:
            if x.dtype == self.dtype: # do nothing
                return x
            
            # this presumes
            # we are given an ndarray
            # with dtype = self.builtin
            # so we typecast, to be safe we make a copy!

            x = np.asarray(x)
            shape = x.shape

            # The last shape entry should match the length
            # of self.dtype

            if x.shape[-1] != len(self.dtype.names):
                warnings.warn("dangerous typecast, shape is unexpected: %d, %d" % (x.shape[-1], len(self.dtype.names)))

            x = np.asarray(x, dtype=self.builtin).ravel()
            y = x.view(self.dtype)
            y.shape = shape[:-1]
            return y
        else:
            if x.dtype == self.builtin: # do nothing
                return x
            y = x.ravel().view(self.builtin)
            y.shape = x.shape + (y.shape[0] / np.product(x.shape),)
            return y

    def reorder(self, name, order=None):
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
        """
        if order is None:
            order = range(len(self.axes))[::-1]
        if name is None:
            name = self.name
        return CoordinateSystem(name, _reorder(self.axes, order))

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
         
    """
    arrays = [np.zeros(2, dtype) for dtype in dtypes]
    notbuiltin = filter(lambda x: not x.dtype.isbuiltin, arrays)
    if notbuiltin:
        raise ValueError('dtypes must be builtin')
    return np.array(arrays).dtype
