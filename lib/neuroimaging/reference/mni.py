from axis import RegularAxis, Axis
from coordinate_system import VoxelCoordinateSystem, DiagonalCoordinateSystem
from mapping import Affine


# MNI template axes
MNI = (
  RegularAxis(name='zspace', length=109, start=-72., step=2.0),
  RegularAxis(name='yspace', length=109, start=-126., step=2.0),
  RegularAxis(name='xspace', length=91, start=-90., step=2.0))

# Default axes
generic = (
  Axis(name='zspace'),
  Axis(name='yspace'),
  Axis(name='xspace'))

# Standard coordinates for MNI template
MNI_voxel = VoxelCoordinateSystem(
  'MNI_voxel', generic, [dim.length for dim in MNI])
MNI_world = DiagonalCoordinateSystem('MNI_world', MNI)

MNI_mapping = Affine(MNI_voxel, MNI_world, MNI_world.transform())

