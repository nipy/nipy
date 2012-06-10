# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pseudo-package for all of the core symbols from the image object and its
reference system.  Use this module for importing core names into your
namespace.

For example:

>>> from nipy.core.api import Image
"""

# Note: The order of imports is important here.
from .reference.coordinate_system import CoordinateSystem
from .reference.coordinate_map import (CoordinateMap, AffineTransform, compose, 
                                       drop_io_dim, append_io_dim)
from .reference.array_coords import Grid, ArrayCoordMap
from .reference.spaces import (vox2scanner, vox2mni, vox2talairach,
                               scanner_space, mni_space, talairach_space)
from .image.image import (Image, fromarray, is_image, subsample, slice_maker,
                          iter_axis, rollaxis as img_rollaxis, rollimg)
from .image.image_spaces import (xyz_affine, is_xyz_affable, as_xyz_image,
                                 make_xyz_image)

from .image.image_list import ImageList

from .utils.generators import (parcels, data_generator, write_data,
                               slice_generator, f_generator,
                               matrix_generator)
