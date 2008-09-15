"""
Coordinate systems are used to represent the spaces in which the images reside.

A coordinate system contains axes; the axes define the coordinates
within the coordinate system.  For example a 3D coordinate system contains 3 axes.

"""

__docformat__ = 'restructuredtext'

import numpy as np

from neuroimaging.core.reference.axis import VoxelAxis
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
        odict.__init__(self, [(ax.name, ax) for ax in axes])


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

    def __setitem__(self, name, value):
        """
        Setting of index values is not allowed.
        
        :Raises TypeError: Always.
        """
        raise TypeError, "CoordinateSystem does not support axis assignment"


    def __eq__(self, other):
        """
        Equality is defined by he axes and the name.

        :Parameters:
            other : `CoordinateSystem`
                The object to be compared with

        :Returns: ``bool``
        """
        return (self.name, self.axes()) == (other.name, other.axes())


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
        return len(self.axes())
    
    def axisnames(self):
        """ A list of the names of the coordinate system's axes. 
        
        :Returns: ``string``
        """
        return self.keys()
        
    def axes(self):
        """ A list of the coordinate system's axes. 
        
        :Returns: ``[`axis.Axis`]``
        """
        return self.values()
    
    def reorder(self, name, order):
        """
        Given a name for the reordered coordinates, and a new order, return a
        reordered coordinate system.

        :Parameters:
            name : ``string``
                The name for the new coordinate system
            order : ``[int]``
                The order of the axes, e.g. [2, 0, 1]

        :Returns:
            `CoordinateSystem`
        """
        if name is None:
            name = self.name
        return CoordinateSystem(name, _reorder(self.axes(), order))


    def reverse(self, name=None):
        """ Create a new coordinate system with the axes reversed. 

        :Parameters:
            name : ``string``
                The name for the new coordinate system

        :Returns: ``CoordinateSystem``
        """
        if name is None:
            name = self.name
        return CoordinateSystem(name, _reverse(self.axes()))


    def hasaxis(self, name):
        """
        Does self contain an axis with the given name

        :Parameters:
            name : ``string``
                The name to be tested for

        :Returns: ``bool``
        """
        return self.has_key(name)


    def getaxis(self, name):
        """ Return the axis with a given name

        :Parameters:
            name : ``string``
                The name of the axis to return

        :Returns: `axis.Axis`
        """
        return self.get(name)


    def isvalidpoint(self, x):
        """
        Verify whether x is a valid coordinate in this system

        :Parameters:
            x : ``tuple`` or ``list`` of ``int`` or ``float``
                A voxel

        :Returns: ``bool``
        """
        return np.all([self.axes()[i].isvalidvalue(x[i])
                       for i in range(self.ndim())])


    def sub_coords(self):
        """
        Return a subset of the coordinate system to be used as part of a subgrid.

        :Returns: ``CoordinateSystem``
        """
        return CoordinateSystem(self.name + "-subgrid", self.axes()[1:])


class VoxelCoordinateSystem(CoordinateSystem):
    """
    Coordinates with a shape -- assumed to be
    voxel coordinates, i.e. if shape = (3, 4, 5) then valid range
    interpreted as [0,2] X [0,3] X [0,4].
    """

    def __init__(self, name, axes, shape=None):
        """
        :Parameters:
            name : ``string``
                The name of the coordinate system
            axes : ``[`axis.Axis`]``
                The axes which make up the coordinate system
            shape : ``tuple`` of ``int``
                The shape of the coordinate system. If ``None`` then the shape
                is determined by the lengths of the ``axes``
        
        :Precondition: ``len(axes) == len(shape)``
        """

        if shape is None:
            self.shape = [dim.length for dim in axes]
        else:
            self.shape = list(shape)
        axes = \
          [VoxelAxis(ax.name, length) for (ax, length) in zip(axes, self.shape)]
        CoordinateSystem.__init__(self, name, axes)


class DiagonalCoordinateSystem(CoordinateSystem):
    """
    Coordinate system with orthogonal axes

    The orthogonality of the axes is assumed by the use of this class
    to define the coordinate system.  Assuming orthogonality allows
    the definition of a method to return an orthogonal transformation
    matrix (tranform method)
    """

    def transform(self):
        """
        Return an orthogonal homogeneous transformation matrix based on the
        step, start attributes of each axis.

        :Returns: ``[[numpy.float]]``
        """
        ndim = self.ndim()
        xform = np.eye((ndim+1))
        for i in range(ndim):
            # Scaling values on diagonal
            xform[i, i] = self.axes()[i].step
            # Translations in last column
            xform[i, -1] = self.axes()[i].start
        return xform

def _reorder(seq, order):
    """ Reorder a sequence. """
    return [seq[i] for i in order]

def _reverse(seq):
    """ Reverse a sequence. """
    return _reorder(seq, range(len(seq)-1, -1, -1))
