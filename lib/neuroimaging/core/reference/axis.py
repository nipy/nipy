"""
The Axis family of classes are used to represent a named axis within a
coordinate system. Axes can be regular discrete or continuous, finite or
infinite.

The current class hirachy looks like this::

                 `Axis`
                   |
        ---------------------------
        |                         |
   `RegularAxis`             `ContinuousAxis`
        |
   `VoxelAxis`


There is currently no support for irregularly spaced axes, however this
could easily be added.
"""

__docformat__ = 'restructuredtext'

import numpy as N

valid = ('time', 'xspace', 'yspace', 'zspace', 'vector_dimension', 'concat')
space = ('zspace', 'yspace', 'xspace')
spacetime = ('time', 'zspace', 'yspace', 'xspace')

class Axis(object):
    """
    This class represents a generic axis. Axes are used in the definition
    of `CoordinateSystem`.
    """

    def __init__(self, name):
        """
        Create an axis with a given name.

        :Parameters:
            `name` : string
                The name for the axis.

        :Precondition: name must be an element of axis.valid
        """
        
        self.name = name
        if self.name not in valid:
            raise ValueError, ('%s is invalid: recognized dimension ' \
                               'names are ' + str(valid)) \
                               % (self.name)

    def __eq__(self, other):
        """ Equality is defined by name.

        :Parameters:
            `other` : `Axis`
                The object to be compared with.

        :Returns:
            `bool`
        """
        return self.name == other.name

    def valid(self, x):
        """ Test if x is a point on the axis. 

        :Parameters:
            `x` : float
                A voxel

        :Returns:
            `bool`
        

        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError

    def max(self):
        """ The maximum value of the axis. 

        :Returns:
            `numpy.float`
        
        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError

    def min(self):
        """ The minimum value of the axis. 

        :Returns:
            `numpy.float`
        
        :Raises NotImplementedError: Abstract method
        """
        raise NotImplementedError

    def range(self):
        """ A (min, max) pair representing the range of the axis. 

        :Returns:
            `(numpy.float, numpy.float)`
        """
        return (self.min(), self.max())


class ContinuousAxis(Axis):
    """
    This class represents an axis which is continuous on some range.
    This range can extend to infinity in either direction.
    """
    def __init__(self, name, low=-N.inf, high=N.inf):
        """

        :Parameters:
            `name` : string
                The name for the axis
            `low` : numpy.float
                The lower bound of the axis
            `high` : numpy.float
                The upper bound of the axis

        Notes
        -----

        If low > high, the behaviour of the class is undefined, though
        this can be changed if a particular choice of behaviour is clearly
        useful.
        
        :Precondition: name must be an element of axis.valid
        """
        self.low = low
        self.high = high
        Axis.__init__(self, name)

    def __eq__(self, other):
        """ Equality is defined by name and range. 

        :Parameters:
            `other` : `ContinuousAxis`
                The object to be compared with

        :Returns:
            `bool`
        """
        return self.range() == other.range() and \
               Axis.__eq__(self, other)

    def valid(self, x):
        """ Test if x is a point on the axis. 

        The axis is defined as the range [low:high).

        :Parameters:
            `x` : float
                A voxel

        :Returns:
            `bool`
        """ 
        return self.low <= x < self.high

    def max(self):
        """ The maximum value of the axis. 

        :Returns:
            `numpy.float`
        """    
        return self.high

    def min(self):
        """ The minimum value of the axis. 

        :Returns:
            `numpy.float`
        """    
        return self.low

            
class RegularAxis (Axis):
    """
    This class represents a regularly spaced axis. Axes are used in the
    definition of Coordinate systems. 

    Example
    -------
    
    >>> from neuroimaging.core.reference.axis import RegularAxis
    >>> from numpy import allclose, array
    >>> r = RegularAxis(name='xspace',length=10, start=0.25, step=0.3)
    >>> allclose(r.values(), array([ 0.25,  0.55,  0.85,  1.15,  1.45,  1.75,  2.05,  2.35,  2.65,  2.95]))
    True
    >>>
    """

    def __init__(self, name, length=N.inf, start=0, step=1):
        """
        Create a regularly spaced axis with a given name.

        :Parameters:
            `name` : string
                The name for the axis
            `length` : numpy.float
                The overall length of the axis
            `start` : numpy.float
                The starting value of the axis
            `step` : numpy.float
                The spacing of the points on the axis

        :Precondition: name must be an element of axis.valid
        """

        self.length = length
        self.start = start
        self.step = step        
        Axis.__init__(self, name)

    def __eq__(self, other):        
        """ Equality is defined by (start, stop, length) and name. 

        :Parameters:
            `other` : `RegurlarAxis`
                The object to be compared with

        :Returns:
            `bool`
        """
        return self.length == other.length and \
               self.start == other.start and \
               self.step == other.step and \
               Axis.__eq__(self, other)

    def valid(self, x):
        """ Test if x is a point on the axis. 

        :Parameters:
            `x` : float
                A voxel

        :Returns:
            `bool`
        """ 
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

        :Returns:        
            `numpy.ndarray(numpy.float)` or `generator` of `numpy.float`
        """
        if self.length == N.inf:
            def generator(x):
                while True:
                    yield x
                    x += self.step
            return generator(self.start)
        else:
            return N.linspace(self.min(), self.max() + self.step, self.length,
                              False)

    def max(self):
        """ The maximum value of the axis. 

        :Returns:
            `numpy.float`
        """    
        return self.start + self.step*(self.length - 1)

    def min(self):
        """ The minimum value of the axis. 

        :Returns:
            `numpy.float`
        """    
        return self.start

class VoxelAxis (RegularAxis):
    """
    A RegularAxis which starts at 0 and has a step of 1.
    """
    def __init__(self, name, length=N.inf):
        """
        Create a voxel axis with a given name.


        :Parameters:
            `name` : string
                The name for the axis
            `length` : numpy.float
                The overall length of the axis

        :Precondition: name must be an element of axis.valid
        """    
        RegularAxis.__init__(self, name, length, start=0, step=1)



generic = (
  ContinuousAxis(name='zspace'),
  ContinuousAxis(name='yspace'),
  ContinuousAxis(name='xspace'))
"""Default axes"""
   

