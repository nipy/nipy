from numpy import zeros, Float

from odict import odict
from attributes import readonly

from neuroimaging import reorder, reverse, hasattrs
from neuroimaging.reference import axis


##############################################################################
class CoordinateSystem(odict):
    "A simple class to carry around coordinate information in one bundle."

    class name (readonly): "coordinate system name"; implements=str
    class axes (readonly):
        "list of axes"; implements=tuple; get=lambda _,s: s.values()
    class axisnames (readonly):
        "list of axis names"; implements=tuple; get=lambda _,s: s.keys()
    class ndim (readonly): "number of dimensions"; get=lambda _,s: len(s.axes)
 
    #-------------------------------------------------------------------------
    def __init__(self, name, axes):
        self.name = name
        self.ndim = len(axes)
        self.axes = tuple(axes)
        odict.__init__(self, [(ax.name,ax) for ax in axes])

    #-------------------------------------------------------------------------
    def __getitem__(self, axisname):
        try:
            return odict.__getitem__(self, axisname)
        except KeyError:
            raise KeyError(
              "axis '%s' not found, names are %s"%(axisname,self.keys()))
        
    #-------------------------------------------------------------------------
    def __setitem__(self, name, value):
        raise TypeError, "CoordinateSystem does not support axis assignment"

    #-------------------------------------------------------------------------
    def __eq__(self, other):
        if not hasattrs(other, "name", "axes"): return False
        return (self.name,self.axes)==(other.name,other.axes)

    #-------------------------------------------------------------------------
    def __str__(self):
        _dict = {'name': self.name, 'axes':self.axes}
        return `_dict`
   
    #-------------------------------------------------------------------------
    def reorder(self, name, order):
        """
        Given a name for the reordered coordinates, and a new order, return a
        reordered coordinate system.
        """
        if name is None: name = self.name
        return CoordinateSystem(name, reorder(axes, order))

    #-------------------------------------------------------------------------
    def reverse(self, name=None):
        if name is None: name = self.name
        return CoordinateSystem(name, reverse(self.axes))

    #-------------------------------------------------------------------------
    def hasaxis(self, name): return self.has_key(name)

    #-------------------------------------------------------------------------
    def getaxis(self, name): return self.get(name)


##############################################################################
class VoxelCoordinateSystem(CoordinateSystem):
    """
    Coordinates with a shape -- assumed to be
    voxel coordinats, i.e. if shape = [3,4,5] then valid range
    interpreted as [0,2] X [0,3] X [0,4].
    """
    class shape (readonly): implements=list

    #-------------------------------------------------------------------------
    def __init__(self, name, axes, shape=None):
        if shape is None:
            try:
                self.shape = [dim.length for dim in axes]
            except:
                raise ValueError, 'must specify a shape or axes must have lengths'
        else:
            self.shape = list(shape)
        CoordinateSystem.__init__(self, name, axes)

    #-------------------------------------------------------------------------
    def isvalid(self, x):
        """
        Verify whether x is a valid (voxel) coordinate.
        Also returns sensible answer for OrthogonalCoordinates.
        """
        if not hasattr(self, '_box'):
            self._getbox()
        test = 1
        for i in range(self.ndim):
            test *= (greater_equal(self.x[i], self._box[i][0]) *
                     less_equal(self.x[i], self._box[i][1]))
        return test

    #-------------------------------------------------------------------------
    def _getbox(self):
        self._box = []
        for i in range(self.ndim):
            dim = self.axes[i]
            try:
                v = dim.values()
            except:
                v = range(self.shape[i])
            self._box.append(min(v), max(v))


##############################################################################
class DiagonalCoordinateSystem(VoxelCoordinateSystem):
    class axes (readonly): default=axis.MNI

    #-------------------------------------------------------------------------
    def __init__(self, name, axes):
        shape = [dim.length for dim in axes]
        VoxelCoordinateSystem.__init__(self, name, axes, shape=shape)
        
    #-------------------------------------------------------------------------
    def transform(self):
        """
        Return an orthogonal  homogeneous transformation matrix based on the
        step, start, length attributes of each axis.
        """
        value = zeros((self.ndim+1,)*2, Float)
        value[self.ndim, self.ndim] = 1.0
        for i in range(self.ndim):
            value[i,i] = self.axes[i].step
            value[i, self.ndim] = self.axes[i].start
        return value

# Standard coordinates for MNI template
MNI_voxel = VoxelCoordinateSystem(
  'MNI_voxel', axis.generic, [dim.length for dim in axis.MNI])
MNI_world = DiagonalCoordinateSystem('MNI_world', axis.MNI)
