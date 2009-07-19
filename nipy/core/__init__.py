"""
Package containing core nipy classes.
"""
__docformat__ = 'restructuredtext'

from .image.base_image import BaseImage
from .image.affine_image import AffineImage
from .transforms.transform import Transform
from .transforms.affine_transform import AffineTransform

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
