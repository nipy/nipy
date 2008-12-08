"""
A set of reference object which represent the MNI space.
"""

__docformat__ = 'restructuredtext'

from neuroimaging.core.reference.axis import RegularAxis, VoxelAxis
from neuroimaging.core.reference.coordinate_system import VoxelCoordinateSystem, \
    StartStepCoordinateSystem
from neuroimaging.core.reference.mapping import Affine



MNI_axes = (
  RegularAxis(name='zspace', start=-72., step=2.0),
  RegularAxis(name='yspace', start=-126., step=2.0),
  RegularAxis(name='xspace', start=-90., step=2.0))
""" The three spatial axes in MNI space """

MNI_voxel = VoxelCoordinateSystem('MNI_voxel', [VoxelAxis(n, length=l) for n, l in zip(['zspace', 'yspace', 'xspace'], (91,109,91))])
""" Standard voxel space coordinate system for MNI template """

MNI_world = StartStepCoordinateSystem('MNI_world', MNI_axes)
""" Standard real space coordinate system for MNI template """

MNI_mapping = Affine(MNI_world.affine)
""" A mapping between the MNI voxel space and the MNI real space """
