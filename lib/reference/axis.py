import numpy as N
import enthought.traits as traits
import UserDict
import tag

valid = ['xspace', 'yspace', 'zspace', 'time', 'vector_dimension']
space = ['zspace', 'yspace', 'xspace']

class Axis(traits.HasTraits):
    """
    This class represents a generic axis. Axes are used
    in the definition Coordinate system.
    """
    
    name = traits.Str()
    tag = traits.Int(tag.new())

    def __init__(self, **extra_args):
        traits.HasTraits.__init__(self, **extra_args)
        if self.name not in valid:
            raise ValueError, 'recognized dimension names are ' + `valid`

    def __eq__(self, dim):
        """
        Verify if two axes are equal by checking tag.
        """
        return self.tag == dim.tag

class VoxelAxis(Axis):
    """
    A axis with a length as well.
    """
    
    length = traits.Int(1)

    def values(self):
        return N.arange(self.length)

class RegularAxis(VoxelAxis):
    """
    This class represents a regularly spaced axis. Axes are used
    in the definition Coordinate system. The attributes step and start
    are usually
    ignored if a valid transformation matrix is provided -- otherwise
    they can be used to create an orthogonal transformation matrix.


    """
    
    step = traits.Float(1.0)
    start = traits.Float()

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

xspace = Axis(name='xspace')
yspace = Axis(name='yspace')
zspace = Axis(name='zspace')
generic = [zspace, yspace, xspace]

# MNI template axes

xspace = Axis(name='xspace', length=91, start=-90., step=2.0)
yspace = Axis(name='yspace', length=109, start=-126., step=2.0)
zspace = Axis(name='zspace', length=109, start=-72., step=2.0)
MNI = [zspace, yspace, xspace]
   
