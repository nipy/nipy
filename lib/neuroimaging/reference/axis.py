import numpy as N
from attributes import readonly

valid = ['xspace', 'yspace', 'zspace', 'time', 'vector_dimension', 'concat']
space = ['zspace', 'yspace', 'xspace']
spacetime = ['time', 'zspace', 'yspace', 'xspace']


class Axis (object):
    """
    This class represents a generic axis. Axes are used in the definition
    of CoordinateSystem.
    """
    class name (readonly): "dimension name"; implements=str


    def __init__(self, name):
        self.name = name
        if self.name not in valid:
            raise ValueError, 'recognized dimension names are ' + `valid`


    def __eq__(self, axis):
        "Equality is defined by name."
        return hasattr(axis,"name") and self.name == axis.name


class VoxelAxis (Axis):
    "An axis with a length as well."

    class length (readonly): "number of voxel positions"; default=1

    def __init__(self, name, length=None):
        Axis.__init__(self, name)
        if length is not None: self.length = length


    def __len__(self): return self.length


    def __eq__(self, axis):
        "Equality is defined by name and length."
        return Axis.__eq__(self, axis) and hasattr(axis,"length") and \
          self.length == axis.length


    def values(self): return N.arange(self.length)



class RegularAxis (VoxelAxis):
    """
    This class represents a regularly spaced axis. Axes are used in the
    definition Coordinate system. The attributes step and start are usually
    ignored if a valid transformation matrix is provided -- otherwise they
    can be used to create an orthogonal transformation matrix.

    >>> r = RegularAxis(name='xspace',length=10, start=0.25, step=0.3)
    >>> r.values()
    array([ 0.25,  0.55,  0.85,  1.15,  1.45,  1.75,  2.05,  2.35,  2.65,  2.95])
    """

    class start (readonly): default=0.
    class step (readonly): default=1.


    def __init__(self, name, length=None, start=None, step=None):
        VoxelAxis.__init__(self, name, length=length)
        if start is not None: self.start = start
        if step is not None: self.step = step


    def __repr__(self):
        return "%s('%s', length=%s, start=%s, step=%s)"%(
          self.__class__.__name__,
          self.name, self.length, self.start, self.step)


    def values(self):
        """
        Return an array of values for the axis, based on step, start, length
        attributes.
        """
        return N.arange(self.start, self.start + self.step*self.length,
          self.step).astype(N.float64)


    def __eq__(self, axis, tol=1.0e-07):
        "Test equality of two axes by name, length, and values."
        if not isinstance(axis, Axis): return False
        v = self.values()
        w = axis.values()
        return VoxelAxis.__eq__(self, axis) and ((v-w)**2).sum()\
               / N.sqrt(((v - v.mean())**2).sum()*((w - w.mean())**2).sum()) < tol

# Default axes
generic = (
  Axis(name='zspace'),
  Axis(name='yspace'),
  Axis(name='xspace'))

# MNI template axes
MNI = (
  RegularAxis(name='zspace', length=109, start=-72., step=2.0),
  RegularAxis(name='yspace', length=109, start=-126., step=2.0),
  RegularAxis(name='xspace', length=91, start=-90., step=2.0))
   

if __name__ == "__main__":
    import doctest
    doctest.testmod()
