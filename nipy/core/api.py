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
from .reference.spaces import (vox2scanner, vox2mni, vox2talairach)
from .image.image import (Image, fromarray, is_image, subsample, slice_maker,
                          iter_axis, rollaxis as img_rollaxis)
from .image.image_spaces import (xyz_affine, is_xyz_affable, as_xyz_affable)
from .image.affine_image import AffineImage

from .image.image_list import ImageList

from .utils.generators import (parcels, data_generator, write_data,
                               slice_generator, f_generator,
                               matrix_generator)

from .image.xyz_image import lps_output_coordnames, ras_output_coordnames
