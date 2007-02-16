"""
Frame of reference/coordinates package.

Mathematical model
==================
The idea of a chart :lm:`\\phi` : I{U} S{sub} I{M} S{->} B{R}^k
on a "manifold" I{M}.  For a chart both input (I{M}) and output coordinates
(B{R}^k) must be defined and a map relating the two coordinate systems.

Description
===========
The modules in this package contains classes which define the space in which an
image exists and also functions for manipulating and traversing this space.

The basic class which defines an image space is a SamplingGrid (grid.py). A 
SamplingGrid consists of an input CoordinateSystem (coordinate_system.py), an
output CoordinateSystem, and a Mapping (mapping.py) which converts point in the
input space to points in the output space.

A `CoordinateSystem` consists of a set of ordered `Axis` (axis.py) objects. Each
Axis can be either discrete (`DiscreteAxis`) or continuous (`ContinuousAxis`). 

The typical use of a `SamplingGrid` is to define how points in an `Image`
(core.image.__init__.py) object's raw data map into real space. 

`Image` traversal is general done in terms of the underlying grid, and a number of
iterators are provided to traverse points in the grid (iterators.py). Access to
available iterators is done through the SamplingGrid interface, rather than 
accessing the iterator classes directly. 

The other common image access method is to take slices through the grid. In 
slices.py functions are presented which will return a `SamplingGrid` representing
a single slice through a larger grid.

"""
__docformat__ = 'restructuredtext'

from neuroimaging.core.reference import \
  axis, coordinate_system, grid, mapping, mni, slices

__all__ = ["axis", "coordinate_system", "grid", "mapping", 
           "mni", "slices"]


