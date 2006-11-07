"""
Coordinate systems are used to represent the spaces in which the images reside.
"""

import numpy as N

from neuroimaging import reorder, reverse
from neuroimaging.core.reference.axis import VoxelAxis
from neuroimaging.utils.odict import odict

class CoordinateSystem(odict):
    "A simple class to carry around coordinate information in one bundle."


    def __init__(self, name, axes):
        """
        Create a coordinate system with a given name and axes.
        
        @param name: The name of the coordinate system
        @type name: C{string}
        @param axes: The axes which make up the coordinate system
        @type axes: C{[L{Axis}]}
        """
        self.name = name
        odict.__init__(self, [(ax.name, ax) for ax in axes])


    def __getitem__(self, axisname):
        """
        Return an axis indexed by name
        
        @param axisname: The name of the axis to return
        @type axisname: C{string}
        @rtype: C{Axis}
        
        @raises KeyError: If axisname is not the name of an axis in this 
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
        
        @raises TypeError: Always.
        """
        raise TypeError, "CoordinateSystem does not support axis assignment"


    def __eq__(self, other):
        """
        Equality is defined by he axes and the name.
        
        @param other: The object to be compared with
        @type other: L{CoordinateSystem}
        @rtype: C{bool}
        """
        return (self.name, self.axes()) == (other.name, other.axes())


    def __str__(self):
        """
        Create a string representation of the coordinate system

        @rtype: C{string}
        """
        _dict = {'name': self.name, 'axes':self.axes}
        return `_dict`
   
    def ndim(self):
        """ Number of dimensions 
        
        @rtype: C{int}
        """
        return len(self.axes())
    
    def axisnames(self):
        """ A list of the names of the coordinate system's axes. 
        
        @rtype: C{[string]}
        """
        return self.keys()
        
    def axes(self):
        """ A list of the coordinate system's axes. 
        
        @rtype: C{[L{Axis}]}
        """
        return self.values()
    
    def reorder(self, name, order):
        """
        Given a name for the reordered coordinates, and a new order, return a
        reordered coordinate system.
        
        @param name: The name for the new coordinate system
        @type name: C{string}
        @param order: The new order of axes, e.g. C{[2,0,1]}
        @type order: C{[int]}
        @rtype: L{CoordinateSystem}
        """
        if name is None:
            name = self.name
        return CoordinateSystem(name, reorder(self.axes(), order))


    def reverse(self, name=None):
        """ Create a new coordinate system with the axes reversed. 
        
        @param name: The name for the new coordinate system
        @type name: C{string}
        @rtype: L{CoordinateSystem}
        """
        if name is None:
            name = self.name
        return CoordinateSystem(name, reverse(self.axes()))


    def hasaxis(self, name):
        """
        Does self contain an axis with the given name
        
        @param name: The name to be tested for
        @type name: C{string}
        
        @rtype: C{bool}
        """
        return self.has_key(name)


    def getaxis(self, name):
        """ Return the axis with a given name

        @param name: The name of the axis to return
        @type name: C{string}
        @rtype: C{Axis}        
        """
        return self.get(name)


    def isvalid(self, x):
        """
        Verify whether x is a valid coordinate.
        
        @param x: a voxel
        @rtype: bool
        """
        return N.all([self.axes()[i].valid(x[i]) for i in range(self.ndim())])


class VoxelCoordinateSystem(CoordinateSystem):
    """
    Coordinates with a shape -- assumed to be
    voxel coordinates, i.e. if shape = [3,4,5] then valid range
    interpreted as [0,2] X [0,3] X [0,4].
    """

    def __init__(self, name, axes, shape=None):
        if shape is None:
            self.shape = [dim.length for dim in axes]
        else:
            self.shape = list(shape)
        axes = \
          [VoxelAxis(ax.name, length) for (ax, length) in zip(axes, self.shape)]
        CoordinateSystem.__init__(self, name, axes)


class DiagonalCoordinateSystem(CoordinateSystem):

    def __init__(self, name, axes):
        self.shape = [dim.length for dim in axes]
        CoordinateSystem.__init__(self, name, axes)
        
    def transform(self):
        """
        Return an orthogonal homogeneous transformation matrix based on the
        step, start, length attributes of each axis.
        
        @rtype: C{[[numpy.float]]}
        """
        value = N.zeros((self.ndim()+1,)*2)
        value[self.ndim(), self.ndim()] = 1.0
        for i in range(self.ndim()):
            value[i,i] = self.axes()[i].step
            value[i, self.ndim()] = self.axes()[i].start
        return value

