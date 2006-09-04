"""
Coordinate systems are used to represent the spaces in which the images reside.
"""

import numpy as N

from neuroimaging import reorder, reverse, hasattrs
from neuroimaging.core.reference.axis import VoxelAxis
from neuroimaging.utils.odict import odict

class CoordinateSystem(odict):
    "A simple class to carry around coordinate information in one bundle."


    def __init__(self, name, axes):
        self.name = name
        odict.__init__(self, [(ax.name, ax) for ax in axes])


    def __getitem__(self, axisname):
        try:
            return odict.__getitem__(self, axisname)
        except KeyError:
            raise KeyError(
              "axis '%s' not found, names are %s"%(axisname,self.keys()))

    def __setitem__(self, name, value):
        raise TypeError, "CoordinateSystem does not support axis assignment"


    def __eq__(self, other):
        if not hasattrs(other, "name", "axes"): return False
        return (self.name, self.axes())==(other.name, other.axes())


    def __str__(self):
        _dict = {'name': self.name, 'axes':self.axes}
        return `_dict`
   
    def ndim(self):
        """ number of dimensions """
        return len(self.axes())
    
    def axisnames(self):
        return self.keys()
        
    def axes(self):
        return self.values()
    
    def reorder(self, name, order):
        """
        Given a name for the reordered coordinates, and a new order, return a
        reordered coordinate system.
        """
        if name is None: name = self.name
        return CoordinateSystem(name, reorder(self.axes(), order))


    def reverse(self, name=None):
        if name is None: name = self.name
        return CoordinateSystem(name, reverse(self.axes()))


    def hasaxis(self, name):
        return self.has_key(name)


    def getaxis(self, name):
        return self.get(name)


    def isvalid(self, x):
        """
        Verify whether x is a valid coordinate.
        """
        return not False in [self.axes()[i].valid(x[i]) for i in range(self.ndim())]


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
        axes = [VoxelAxis(ax.name, length) for (ax, length) in zip(axes, self.shape)]
        CoordinateSystem.__init__(self, name, axes)


class DiagonalCoordinateSystem(CoordinateSystem):

    def __init__(self, name, axes):
        self.shape = [dim.length for dim in axes]
        CoordinateSystem.__init__(self, name, axes)
        
    def transform(self):
        """
        Return an orthogonal homogeneous transformation matrix based on the
        step, start, length attributes of each axis.
        """
        value = N.zeros((self.ndim()+1,)*2, N.float64)
        value[self.ndim(), self.ndim()] = 1.0
        for i in range(self.ndim()):
            value[i,i] = self.axes()[i].step
            value[i, self.ndim()] = self.axes()[i].start
        return value

