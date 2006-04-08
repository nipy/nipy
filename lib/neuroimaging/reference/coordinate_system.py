import numpy as N
import enthought.traits as traits
import UserDict
import axis
import uuid

class CoordinateSystem(traits.HasTraits,UserDict.DictMixin):
    """A simple class to carry around coordinate information in one bundle.

    """

    name = traits.Str()
    ndim = traits.Int(3)
    axes = traits.List(axis.generic)
    tag = traits.Trait(uuid.Uuid())
    axisnames = traits.ListStr()
    
    def hasaxis(self, name):
        for axis in self.axes:
            if axis.name == name:
                return True
        return False

    def getaxis(self, name):
        if self.hasaxis(name):
            for axis in self.axes:
                if axis.name == name:
                    return axis
        return False

    def __getitem__(self, axisname):
        if axisname not in self.axisnames:
            raise KeyError, 'no such axis'
        else:
            which = self.axisnames.index(axisname)
            return self.axes[which]
        
    def __setitem__(self, any):
        raise TypeError, "object doesn't support item assignment"

    def keys(self):
        return self.axisnames

    def __init__(self, name, axes):
        self.name = name
        self.ndim = len(axes)
        self.axes = list(axes)
        self.axisnames += [axes[x].name for x in range(len(axes))]

    def __eq__(self, other):
        test = (self.axes == other.axes)
        return test
    
    def reorder(self, name, order):
        """Given a name for the reordered coordinates, and a new order, return a reordered coordinate system."""
        axes = []
        for coord in order:
            axes.append(self.axes[coord])
        return CoordinateSystem(name, axes)

    def __str__(self):
        _dict = {'name': self.name, 'axes':self.axes}
        return `_dict`

    def __eq__(self, other):
        try:
            name_test = (self.name == other.name)
            dim_test = N.product(map(lambda x: self.axes[x] == other.axes[x], range(len(self.axes))))
            ndim_test = (len(self.axes) == len(other.axes))
            return name_test * dim_test * ndim_test
        except:
            return False

class VoxelCoordinateSystem(CoordinateSystem):
    """
    Coordinates with a shape -- assumed to be
    voxel coordinats, i.e. if shape = [3,4,5] then valid range
    interpreted as [0,2] X [0,3] X [0,4].
    """

    shape = traits.ListInt()

    def __init__(self, name, axes, shape=None):
        if shape is None:
            try:
                self.shape = [dim.length for dim in axes]
            except:
                raise ValueError, 'must specify a shape or axes must have lengths'
        else:
            self.shape = list(shape)
            
        CoordinateSystem.__init__(self, name, axes)

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

    def _getbox(self):
        self._box = []
        for i in range(self.ndim):
            dim = self.axes[i]
            try:
                v = dim.values()
            except:
                v = range(self.shape[i])
            self._box.append(min(v), max(v))

class DiagonalCoordinateSystem(VoxelCoordinateSystem):

    axes = traits.List(axis.MNI)

    def __init__(self, name, axes):
        shape = [dim.length for dim in axes]
        VoxelCoordinateSystem.__init__(self, name, axes, shape=shape)
        
    def transform(self):
        """
        Return an orthogonal  homogeneous transformation matrix based on the
        step, start, length attributes of each axis.
        """

        value = N.zeros((self.ndim+1,)*2, N.Float)
        value[self.ndim, self.ndim] = 1.0
        for i in range(self.ndim):
            value[i,i] = self.axes[i].step
            value[i, self.ndim] = self.axes[i].start
        return value

# Standard coordinates for MNI template

MNI_voxel = VoxelCoordinateSystem('MNI_voxel', axis.generic, [dim.length for dim in axis.MNI])

MNI_world = DiagonalCoordinateSystem('MNI_world', axis.MNI)
