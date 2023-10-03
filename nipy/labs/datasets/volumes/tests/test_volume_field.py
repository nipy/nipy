# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Testing data image interface.
"""

import pytest

from ...transforms.transform import CompositionError, Transform

# Local imports
from ..volume_field import VolumeField


################################################################################
# Tests
def test_interface():
    img = VolumeField()
    img.world_space = 'world'
    for method in ('get_transform', 'as_volume_img'):
        method = getattr(img, method)
        assert pytest.raises(NotImplementedError, method)

    assert pytest.raises(CompositionError, img.composed_with_transform, Transform('world2', 'world', mapping=map))

    assert pytest.raises(NotImplementedError, img.composed_with_transform, Transform('world', 'world2', mapping=map))

    assert pytest.raises(NotImplementedError, img.resampled_to_img, None)

    assert pytest.raises(NotImplementedError, img.values_in_world, None, None, None)
