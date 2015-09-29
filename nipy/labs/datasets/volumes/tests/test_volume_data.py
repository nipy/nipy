# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Testing volume data interface.
"""
from __future__ import absolute_import

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

