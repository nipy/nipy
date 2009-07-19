"""
The Image class provides the interface which should be used
by users at the application level. The image provides a coordinate map,
and the data itself.

"""
__docformat__ = 'restructuredtext'

import image, roi, generators
from image import Image

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench

