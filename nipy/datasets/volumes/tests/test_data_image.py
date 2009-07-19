"""
Testing data image interface.
"""

import nose
import copy

# Local imports
from ..data_image import DataImage

################################################################################
# Tests
def test_data_image():
    """ Sanity testing of the data image class.
    """
    img = DataImage()
    # Test that the repr doesn't raise an error
    yield repr, img

    # Check the non-implemented interface
    yield nose.tools.assert_raises, NotImplementedError, \
                        img.values_in_world, 0, 0, 0

    yield nose.tools.assert_raises, NotImplementedError, \
                        img.as_volume_img

    yield nose.tools.assert_raises, NotImplementedError, copy.copy, img

