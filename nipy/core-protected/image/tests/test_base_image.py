"""
Testing base image interface.
"""

import numpy as np

# Local imports
from neuroimaging.core.image.base_image import BaseImage

def test_base_image():
    """ Sanity testing of the base image class.
    """
    img = BaseImage()
    # Test that the repr doesn't raise an error
    yield repr, img
    # Check that values_in_world and resampled_to_affine indirectly raises
    # NotImplementedError
    yield np.testing.assert_raises, NotImplementedError, \
                        img.values_in_world, 0, 0, 0
    yield np.testing.assert_raises, NotImplementedError, \
                        img.resampled_to_affine
    yield np.testing.assert_raises, NotImplementedError, \
                        img.resampled_to_img, img


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

