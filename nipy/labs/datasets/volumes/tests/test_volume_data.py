# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Testing volume data interface.
"""

import copy

import pytest

# Local imports
from ..volume_data import VolumeData


################################################################################
# Tests
def test_volume_data():
    """ Sanity testing of the VolumeData class.
    """
    vol = VolumeData()
    # Test that the repr doesn't raise an error
    repr(vol)

    # Check the non-implemented interface
    pytest.raises(NotImplementedError,
                        vol.values_in_world, 0, 0, 0)

    pytest.raises(NotImplementedError,
                        vol.as_volume_img)

    pytest.raises(NotImplementedError, copy.copy, vol)
