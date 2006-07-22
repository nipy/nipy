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

    def __init__(self, name, values=None):
        self.name = name
        if self.name not in valid:
            raise ValueError, 'recognized dimension names are ' + `valid`
        if values is None:
            self._values = []
        else:
            self._values = values
        self.length = len(self._values)

    def values(self):
        return self._values

    def __eq__(self, axis):
        "Equality is defined by name and values."
        return hasattr(axis,"name") and self.name == axis.name and self.length == axis.length and N.all(N.equal(self.values(), axis.values()))
    
    def __len__(self):
        return self.length

class RegularAxis (Axis):
    """
    This class represents a regularly spaced axis. Axes are used in the
    definition Coordinate system. The attributes step and start are usually
    ignored if a valid transformation matrix is provided -- otherwise they
    can be used to create an orthogonal transformation matrix.

    >>> from neuroimaging.reference.axis import RegularAxis
    >>> from numpy import allclose, array
    >>> r = RegularAxis(name='xspace',length=10, start=0.25, step=0.3)
    >>> allclose(r.values(), array([ 0.25,  0.55,  0.85,  1.15,  1.45,  1.75,  2.05,  2.35,  2.65,  2.95]))
    True
    >>>
    """

    def __init__(self, name, length=0, start=0, step=0):
        _values = N.linspace(start, start + step*length, length, False)
        Axis.__init__(self, name, _values)
        self.start = start
        self.step = step



class VoxelAxis (RegularAxis):
    "An axis with a length as well."

    def __init__(self, name, length=0):
        RegularAxis.__init__(self, name, length, start=0, step=1)


# Default axes
generic = (
  Axis(name='zspace'),
  Axis(name='yspace'),
  Axis(name='xspace'))

   

if __name__ == "__main__":
    import doctest
    doctest.testmod()
