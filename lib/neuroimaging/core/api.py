"""
Pseudo-package for all of the core symbols from the image object and its reference
system.  Use this module for importing core names into your namespace. For example:
 from neuorimaging.core.api import Image
"""

# Note: The order of imports is important here.
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.core.image.image import Image


from neuroimaging.core.image.iterators import SliceIterator, \
     SliceParcelIterator, ParcelIterator, fMRIParcelIterator, \
     fMRISliceParcelIterator, ImageSequenceIterator
from neuroimaging.core.reference.mapping import Mapping, Affine
from neuroimaging.core.reference.coordinate_system import CoordinateSystem
