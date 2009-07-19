"""
Testing base image interface.
"""

import numpy as np

# Local imports
from ..base_image import BaseImage
from ...transforms.transform import Transform

def mapping(x, y, z):
    return 2*x, y, 0.5*z

def id(x, y, z):
    return x, y, z

def test_base_image():
    """ Sanity testing of the base image class.
    """
    transform = Transform('voxels', 'world', mapping)
    img = BaseImage(data=np.random.random((10, 10, 10)),
                    transform=transform,
                    )
    # Test that the repr doesn't raise an error
    yield repr, img

    # We cannot calculate the values in the world, because the transform 
    # is not invertible.
    
    yield np.testing.assert_raises, ValueError, \
                        img.values_in_world, 0, 0, 0
    yield np.testing.assert_raises, NotImplementedError, \
                        img.resampled_to_grid
    yield np.testing.assert_raises, NotImplementedError, \
                        img.get_grid



def test_trivial_image():
    """ Test resampling for an image embedded in world space with an
        identity transform. 
    """
    N = 10
    identity = Transform('voxels', 'world', id, id)
    data = np.random.random((N, N, N))
    img = BaseImage(data=data,
                    transform=identity,
                    )
    x, y, z = np.random.random_integers(N, size=(3, 10)) - 1
    data_ = img. values_in_world(x, y, z)
    yield np.testing.assert_almost_equal, data[x, y, z], data_



if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

