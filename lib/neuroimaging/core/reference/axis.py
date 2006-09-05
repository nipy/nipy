"""
The Axis family of classes are used to represent a named axis within a
coordinate system. Axes can be discrete or continuous, finite or infinite.

The current class hirachy looks like this.

                 Axis
                   |
        ---------------------------
        |                         |
        DiscreteAxis        ContinuousAxis
        |   
   RegularAxis
        |
   VoxelAxis


There is currently no support for irregularly spaced axes, however this
could easily be added as a subclass of DiscreteAxis if required.

"""

import numpy as N

valid = ('time', 'xspace', 'yspace', 'zspace', 'vector_dimension', 'concat')
space = ('zspace', 'yspace', 'xspace')
spacetime = ('time', 'zspace', 'yspace', 'xspace')

class Axis(object):
    """
    This class represents a generic axis. Axes are used in the definition
    of CoordinateSystem.
    """

    def __init__(self, name):
        self.name = name
        if self.name not in valid:
            raise ValueError, ('%s is invalid: recognized dimension names are ' + `valid`) \
                % (self.name)

    def __eq__(self, other):
        """ Equality is defined by name """
        return self.name == other.name

    def valid(self, x):
        """ test if x is a point on the axis. """
        raise NotImplementedError

    def max(self):
        """ The maximum value of the axis. """
        raise NotImplementedError

    def min(self):
        """ The minimum value of the axis. """
        raise NotImplementedError

    def range(self):
        """ A (min, max) pair representing the range of the axis. """
        return (self.min(), self.max())


class ContinuousAxis(Axis):
    """
    This class represents an axis which is continuous on some range.
    This range can extend to infinity in either direction.
    """
    def __init__(self, name, low=-N.inf, high=N.inf):
        self.low = low
        self.high = high
        Axis.__init__(self, name)

    def __eq__(self, other):
        """ Equality is defined by name and range. """
        return self.range() == other.range() and \
               Axis.__eq__(self, other)

    def valid(self, x):
        """ The axis is defined as the range [low:high) """
        return self.low <= x < self.high

    def max(self):
        return self.high

    def min(self):
        return self.low

            
class RegularAxis (Axis):
    """
    This class represents a regularly spaced axis. Axes are used in the
    definition of Coordinate systems. The attributes step and start are usually
    ignored if a valid transformation matrix is provided -- otherwise they
    can be used to create an orthogonal transformation matrix.

    >>> from neuroimaging.core.reference.axis import RegularAxis
    >>> from numpy import allclose, array
    >>> r = RegularAxis(name='xspace',length=10, start=0.25, step=0.3)
    >>> allclose(r.values(), array([ 0.25,  0.55,  0.85,  1.15,  1.45,  1.75,  2.05,  2.35,  2.65,  2.95]))
    True
    >>>
    """

    def __init__(self, name, length=N.inf, start=0, step=1):
        self.length = length
        self.start = start
        self.step = step        
        Axis.__init__(self, name)

    def __eq__(self, other):        
        """ Equality is defined by (start, stop, length) and name. """
        return self.length == other.length and \
               self.start == other.start and \
               self.step == other.step and \
               Axis.__eq__(self, other)

    def valid(self, x):
        if x in (-N.inf, N.inf):
            return False
        if self.length == N.inf:
            tmp = (x - self.start)/self.step
            return (tmp == int(tmp)) and (0 <= tmp < self.length)
        else:
            return x in self.values()

    def values(self):
        """
        Return all the values in the axis.
        Return a generator for the infinite case
        """
        if self.length == N.inf:
            def f(x):
                while True:
                    yield x
                    x += self.step
            return f(self.start)
        else:
            return N.linspace(self.min(), self.max() + self.step, self.length, False)

    def max(self):
        return self.start + self.step*(self.length - 1)

    def min(self):
        return self.start

class VoxelAxis (RegularAxis):
    """
    A RegularAxis which starts at 0 and has a step of 1.
    """
    def __init__(self, name, length=N.inf):
        RegularAxis.__init__(self, name, length, start=0, step=1)


# Default axes
generic = (
  ContinuousAxis(name='zspace'),
  ContinuousAxis(name='yspace'),
  ContinuousAxis(name='xspace'))

   

