"""
Coordinate systems are used to represent the spaces in which the images reside.

A coordinate system contains axes; the axes define the coordinates
within the coordinate system.  For example a 3D coordinate system contains 3 axes.

"""

__docformat__ = 'restructuredtext'

import copy, warnings

import numpy as np

from neuroimaging.core.reference.axis import Axis
from neuroimaging.utils.odict import odict

class CoordinateSystem(odict):
    """A simple class to carry around coordinate information in one bundle."""

    def __init__(self, name, axes):
        """
        Create a coordinate system with a given name and axes.

        :Parameters:
            name : ``string``
                The name of the coordinate system
            axes : ``[`axis.Axis`]``
                The axes which make up the coordinate system
        """
        self.name = name
        if len(set([ax.name for ax in axes])) != len(axes):
            raise ValueError, 'axes must have distinct names'
        odict.__init__(self, [(ax.name, ax) for ax in axes])

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
                raise ValueError('could not work out a builtin dtype for this coordinate system')
    builtin = property(_getbuiltin)

    def __getitem__(self, axisname):
        """
        Return an axis indexed by name

        :Parameters:
            axisname : ``string``
                The name of the axis to return

        :Returns: `axis.Axis`
        
        :Raises KeyError: If axisname is not the name of an axis in this 
            coordinate system.
        """
        try:
            return odict.__getitem__(self, axisname)
        except KeyError:
            raise KeyError(
              "axis '%s' not found, names are %s"%(axisname,self.keys()))

    def rename(self, **kwargs):
        """
        Return a new CoordinateSystem with the values renamed.

        >>> axes = [Axis(n) for n in 'abc']
        >>> coords = CoordinateSystem('input', axes)
        >>> print coords.rename(a='x')
        {'axes': [<Axis:"x", dtype=[('x', '<f8')]>, <Axis:"b", dtype=[('b', '<f8')]>, <Axis:"c", dtype=[('c', '<f8')]>], 'name': 'input-renamed'}
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

        :Parameters:
            other : `CoordinateSystem`
                The object to be compared with

        :Returns: ``bool``
        """
        return self.dtype == other.dtype


    def __str__(self):
        """
        Create a string representation of the coordinate system

        :Returns: ``string``
        """
        _dict = {'name': self.name, 'axes':self.axes}
        return `_dict`
   
    def ndim(self):
        """ Number of dimensions 
        
        :Returns: ``int``
        """
        return len(self.axes)
    
    def typecast(self, x, dtype=None):
        """
        Try to safely typecast x into
        an ndarray with a numpy builtin dtype
        with the correct shape, or
        typecast it as an ndarray with self.dtype.

        TODO: Ensure that we can always find a builtin. This
              means modification of the dtypes of all the Axes
              at construction time.
     
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

            x = np.array(x, dtype=self.builtin).ravel()

            # The last shape entry should match the length
            # of self.dtype

            if x.shape[-1] != len(self.dtype.names):
                warnings.warn("dangerous typecast, shape is unexpected: %d, %d" % (x.shape[-1], len(self.dtype.names)))
            y = x.view(self.dtype)
            return y
        else:
            if x.dtype == self.builtin: # do nothing
                return x
            y = x.view(self.builtin)
            return y

    def _getaxisnames(self):
        """ A list of the names of the coordinate system's axes. 
        
        :Returns: ``string``
        """
        return self.keys()
    axisnames = property(_getaxisnames)
        
    def _getaxes(self):
        """ A list of the coordinate system's axes. 
        
        :Returns: ``[`axis.Axis`]``
        """
        return self.values()
    axes = property(_getaxes)

    def reorder(self, name, order=None):
        """
        Given a name for the reordered coordinates, and a new order, return a
        reordered coordinate system. Defaults to reversal.

        :Parameters:
            name : ``string``
                The name for the new coordinate system
            order : ``[int]``
                The order of the axes, e.g. [2, 0, 1]

        :Returns:
            `CoordinateSystem`
        """
        if order is None:
            order = range(len(self.axes))[::-1]
        if name is None:
            name = self.name
        return CoordinateSystem(name, _reorder(self.axes, order))

def _reorder(seq, order):
    """ Reorder a sequence. """
    return [seq[i] for i in order]

def _reverse(seq):
    """ Reverse a sequence. """
    return _reorder(seq, range(len(seq)-1, -1, -1))
