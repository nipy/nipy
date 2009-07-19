"""
Package containing core nipy classes.
"""
__docformat__ = 'restructuredtext'

from .image.base_image import BaseImage
from .image.volume_image import VolumeImage
from .transforms.transform import Transform
from .transforms.affine_transform import AffineTransform

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
