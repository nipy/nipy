"""
Pseudo-package for all of the core symbols from the image object and its reference
system.  Use this module for importing core names into your namespace. For example:
 from neuorimaging.core.api import Image
"""

# Note: The order of imports is important here.
from nipy.core.reference.coordinate_system import CoordinateSystem
from nipy.core.reference.coordinate_map import CoordinateMap, Affine, compose
from nipy.core.reference.array_coords import Grid, ArrayCoordMap

from nipy.core.image.image import Image, merge_images, fromarray

from nipy.core.image.image_list import ImageList

from nipy.core.image.generators import parcels, data_generator, write_data, slice_generator, f_generator, matrix_generator
