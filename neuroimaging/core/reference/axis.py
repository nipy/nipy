"""
FIXME: this docstring is obsolete -- only the Axis class remains

The Axis family of classes are used to represent a named axis within a
coordinate system. Axes can be regular discrete or continuous, finite or
infinite.

The current class hierarchy looks like this::

                 `Axis`
                   |
        ---------------------------
        |                         |
   `RegularAxis`             `ContinuousAxis`
        |
   `VoxelAxis`


There is currently no support for irregularly spaced axes, however
this could easily be added.
"""
import numpy as np

__docformat__ = 'restructuredtext'

