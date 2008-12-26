"""
A set of reference object which represent the MNI space.
"""
from neuroimaging.core.reference.coordinate_system import CoordinateSystem
from neuroimaging.core.reference.coordinate_map import Affine

__docformat__ = 'restructuredtext'


MNI_mapping = Affine.from_start_step(['xspace', 'yspace', 'zspace'],
                                     ['xspace', 'yspace', 'zspace'],
                                     [-72,-126,-90], [2,2,2]) 
MNI_voxel = MNI_mapping.input_coords
MNI_world = MNI_mapping.output_coords

""" A mapping between the MNI voxel space and the MNI real space """
