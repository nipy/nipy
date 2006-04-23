import numpy as N
import enthought.traits as traits
import uuid

valid = ['xspace', 'yspace', 'zspace', 'time', 'vector_dimension', 'concat']
space = ['zspace', 'yspace', 'xspace']
spacetime = ['time', 'zspace', 'yspace', 'xspace']

##############################################################################
class Axis(traits.HasTraits):
    """
    This class represents a generic axis. Axes are used
    in the definition Coordinate system.
    """
    
    name = traits.Str()

    def __init__(self, name):
        traits.HasTraits.__init__(self)
        self.name = name
        self.tag = uuid.Uuid()
        if self.name not in valid:
            raise ValueError, 'recognized dimension names are ' + `valid`

    def __eq__(self, dim):
        """
        Verify if two axes are equal by checking tag.
        """
        return hasattr(self,"tag") and hasattr(dim,"tag") and \
          self.tag == dim.tag


##############################################################################
class VoxelAxis(Axis):
    "A axis with a length as well."
    length = traits.Int(1)

    def __init__(self, name, length=None):
        Axis.__init__(self, name)
        if length is not None: self.length = length

    def values(self): return N.arange(self.length)


##############################################################################
class RegularAxis(VoxelAxis):
    """
    This class represents a regularly spaced axis. Axes are used
    in the definition Coordinate system. The attributes step and start
    are usually ignored if a valid transformation matrix is provided --
    otherwise they can be used to create an orthogonal transformation matrix.

    >>> r = RegularAxis(name='xspace',length=10,start=0.25,step=0.3)
    >>> r.values()
    array([ 0.25,  0.55,  0.85,  1.15,  1.45,  1.75,  2.05,  2.35,  2.65,  2.95])
    """
    step = traits.Float(1.0)
    start = traits.Float()

    def __init__(self, name, length=None, start=None, step=None):
        VoxelAxis.__init__(self, name, length=length)
        if start is not None: self.start = start
        if step is not None: self.step = step

    def __len__(self):
        return self.length
    
    def __repr__(self):
        _dict = {'length':self.length, 'step': self.step, 'start': self.start, 'name': self.name}
        return `_dict`

    def values(self):
        """
        Return an array of values for the axis, based on step, start, length attributes.
        """
        return N.arange(self.start, self.start + self.step * self.length, self.step).astype(N.Float)

    def __eq__(self, dim, tol=1.0e-07):
        """
        Verify if two axes are equal by checking tag.
        """
        v = self.values()
        w = dim.values()
        if N.add.reduce((v-w)**2) / N.sqrt(N.add.reduce((v - N.mean(v))**2) * N.add.reduce((w - N.mean(w))**2)) < tol:
            return True
        else:
            return False

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
