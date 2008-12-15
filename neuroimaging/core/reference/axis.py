"""
FIXME: this docstring is obsolete -- only the Axis class remains

The Axis family of classes are used to represent a named axis within a
coordinate system. Axes can be regular discrete or continuous, finite or
infinite.

The current class hierarchy looks like this::

                 `Axis`
                   |
        ---------------------------
        |                         |
   `RegularAxis`             `ContinuousAxis`
        |
   `VoxelAxis`


There is currently no support for irregularly spaced axes, however
this could easily be added.
"""
import numpy as np


__docformat__ = 'restructuredtext'

class Axis(object):
    """
    This class represents a generic axis. Axes are used in the definition
    of ``CoordinateSystem``.
    """

    def __str__(self):
        return '<Axis:"%(name)s", dtype=%(dtype)s>' % {'name':self.name, 'dtype':`self.dtype.descr`}

    def __len__(self):
        if self._length > 0:
            return self._length
        else:
            raise ValueError, '%s has no length' % `self`

    def __repr__(self):
        return self.__str__()

    def __init__(self, name, dtype=np.float, length=None):
        """
        Create an axis with a given name.

        :Parameters:
            name : ``string``
                The name for the axis.
        """
        self.name = name
        self._dtype = np.dtype(dtype)
        self._length = length

        # verify that the dtype is builtin for sanity
        if not self._dtype.isbuiltin:
            raise ValueError, 'Axis dtypes should be numpy builtin dtypes'

    def _getdtype(self):
        return np.dtype([(self.name, self._dtype)])
    def _setdtype(self, dtype):
        self._dtype = dtype
    dtype = property(_getdtype, _setdtype)

    def _getbuiltin(self):
        return self._dtype
    builtin = property(_getbuiltin)

    def __eq__(self, other):
        """ Equality is defined by dtype.

        :Parameters:
            other : `Axis`
                The object to be compared with.

        :Returns: ``bool``
        """
        return self.dtype == other.dtype


