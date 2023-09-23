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
from .image.image import (
    Image,
    fromarray,
    is_image,
    iter_axis,
    rollimg,
    slice_maker,
    subsample,
)
from .image.image import rollaxis as img_rollaxis
from .image.image_list import ImageList
from .image.image_spaces import as_xyz_image, is_xyz_affable, make_xyz_image, xyz_affine
from .reference.array_coords import ArrayCoordMap, Grid
from .reference.coordinate_map import (
    AffineTransform,
    CoordinateMap,
    append_io_dim,
    compose,
    drop_io_dim,
)
from .reference.coordinate_system import CoordinateSystem
from .reference.spaces import (
    mni_space,
    scanner_space,
    talairach_space,
    vox2mni,
    vox2scanner,
    vox2talairach,
)
from .utils.generators import (
    data_generator,
    f_generator,
    matrix_generator,
    parcels,
    slice_generator,
    write_data,
)
