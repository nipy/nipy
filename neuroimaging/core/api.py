"""
Pseudo-package for all of the core symbols from the image object and its reference
system.  Use this module for importing core names into your namespace. For example:
 from neuorimaging.core.api import Image
"""

# Note: The order of imports is important here.
from neuroimaging.core.image.base_image import BaseImage
from neuroimaging.core.reference.grid import SamplingGrid

from neuroimaging.core.image.iterators import SliceParcelIterator, \
     ParcelIterator, ImageSequenceIterator
from neuroimaging.core.reference.mapping import Mapping, Affine
from neuroimaging.core.reference.coordinate_system import CoordinateSystem, \
     DiagonalCoordinateSystem
from neuroimaging.core.reference.axis import VoxelAxis, space, spacetime

from neuroimaging.core.image.image import Image, merge_images, merge_to_array, zeros
from neuroimaging.core.image.image import load as load_image
from neuroimaging.core.image.image import save as save_image
from neuroimaging.core.image.image import slice_iterator, slice_parcel_iterator, parcel_iterator, fromarray

from neuroimaging.core.image.image_list import ImageList


