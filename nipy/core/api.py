"""
Pseudo-package for all of the core symbols from the image object and its
reference system.  Use this module for importing core names into your
namespace. For example: from nipy.core.api import AffineImage 
"""


from .image.affine_image import AffineImage
from .transforms.affine_transform import AffineTransform
from .transforms.transform import Transform

