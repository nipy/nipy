# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Testing data image interface.
"""
from __future__ import absolute_import

import numpy as np

# Local imports
from ..volume_field import VolumeField
from ...transforms.transform import Transform, CompositionError

################################################################################
# Tests
def test_interface():
    img = VolumeField()
    img.world_space = 'world'
    for method in ('get_transform', 'as_volume_img'): 
        method = getattr(img, method)
        yield np.testing.assert_raises, NotImplementedError, method 

    yield np.testing.assert_raises, CompositionError, \
                    img.composed_with_transform, \
                    Transform('world2', 'world', mapping=map)

    yield np.testing.assert_raises, NotImplementedError, \
                    img.composed_with_transform, \
                    Transform('world', 'world2', mapping=map)

    yield np.testing.assert_raises, NotImplementedError, \
                    img.resampled_to_img, None

    yield np.testing.assert_raises, NotImplementedError, \
                    img.values_in_world, None, None, None


