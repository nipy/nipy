"""
A set of reference object which represent the MNI space.
"""

from neuroimaging.core.reference.axis import RegularAxis, generic
from neuroimaging.core.reference.coordinate_system import VoxelCoordinateSystem, \
    DiagonalCoordinateSystem
from neuroimaging.core.reference.mapping import Affine


# MNI template axes
MNI_axes = (
  RegularAxis(name='zspace', length=109, start=-72., step=2.0),
  RegularAxis(name='yspace', length=109, start=-126., step=2.0),
  RegularAxis(name='xspace', length=91, start=-90., step=2.0))


# Standard coordinates for MNI template
MNI_voxel = VoxelCoordinateSystem(
  'MNI_voxel', generic, [dim.length for dim in MNI_axes])
MNI_world = DiagonalCoordinateSystem('MNI_world', MNI_axes)

MNI_mapping = Affine(MNI_voxel, MNI_world, MNI_world.transform())
""" A mapping between the MNI voxel space and the MNI real space """
