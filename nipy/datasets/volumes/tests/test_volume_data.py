"""
Testing volume data interface.
"""

import nose
import copy

# Local imports
from ..volume_data import VolumeData

################################################################################
# Tests
def test_volume_data():
    """ Sanity testing of the VolumeData class.
    """
    vol = VolumeData()
    # Test that the repr doesn't raise an error
    yield repr, vol

    # Check the non-implemented interface
    yield nose.tools.assert_raises, NotImplementedError, \
                        vol.values_in_world, 0, 0, 0

    yield nose.tools.assert_raises, NotImplementedError, \
                        vol.as_volume_img

    yield nose.tools.assert_raises, NotImplementedError, copy.copy, vol

