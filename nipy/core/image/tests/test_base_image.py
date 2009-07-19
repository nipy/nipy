"""
Testing data image interface.
"""

import nose
import copy

import numpy as np

# Local imports
from ..data_image import DataImage, CompositionError
from ..xyz_image import XYZImage
from ...transforms.transform import Transform

def mapping(x, y, z):
    return 2*x, y, 0.5*z

def inverse_mapping(x, y, z):
    return 0.5*x, y, 2*z

def id(x, y, z):
    return x, y, z

################################################################################
# Tests
def test_constructor():
    yield np.testing.assert_raises, ValueError, DataImage, None, \
        None, {}, 'e'


def test_data_image():
    """ Sanity testing of the data image class.
    """
    transform = Transform('voxels', 'world', mapping)
    img = DataImage(data=np.random.random((10, 10, 10)),
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

    yield nose.tools.assert_equal, img, copy.copy(img)


def test_trivial_image():
    """ Test resampling for an image embedded in world space with an
        identity transform. 
    """
    N = 10
    identity = Transform('voxels', 'world', id, id)
    data = np.random.random((N, N, N))
    img = DataImage(data=data,
                    transform=identity,
                    )
    x, y, z = np.random.random_integers(N, size=(3, 10)) - 1
    data_ = img. values_in_world(x, y, z)
    # Check that passing in arrays with different shapes raises an error
    yield np.testing.assert_raises, ValueError, \
        img.values_in_world, x, y, z[:-1]
    # Check that passing in wrong interpolation keyword raises an error
    yield np.testing.assert_raises, ValueError, \
                        img.values_in_world, 0, 0, 0, 'e'
    yield np.testing.assert_almost_equal, data[x, y, z], data_


def test_transformation():
    """ Test transforming images.
    """
    N = 10
    v2w_mapping = Transform('voxels', 'world1', mapping, 
                            inverse_mapping)
    identity  = Transform('world1', 'world2', id, id) 
    data = np.random.random((N, N, N))
    img1 = DataImage(data=data,
                     transform=v2w_mapping,
                     )
    img2 = img1.transformed_with(identity)
    
    yield nose.tools.assert_equal, img2.world_space, 'world2'

    x, y, z = N*np.random.random(size=(3, 10))
    yield np.testing.assert_almost_equal, img1.values_in_world(x, y, z), \
        img2.values_in_world(x, y, z)

    yield nose.tools.assert_raises, CompositionError, img1.transformed_with, \
            identity.get_inverse()

    yield nose.tools.assert_raises, CompositionError, img1.resampled_to_img, \
            img2
    
    # Resample an image on itself: it shouldn't change much:
    img  = img1.resampled_to_img(img1)
    yield np.testing.assert_almost_equal, data, img.get_data()

    # Check that if I 'resampled_to_img' on an XYZImage, I get an
    # XYZImage, and vice versa 
    xyz_image = XYZImage(data, np.eye(4), 'world')
    identity  = Transform('voxels', 'world', id, id) 
    image = DataImage(data, identity)
    image2 = image.resampled_to_img(xyz_image)
    yield nose.tools.assert_true, isinstance(image2, XYZImage)
    xyz_image2 = xyz_image.resampled_to_img(image)
    yield nose.tools.assert_true, isinstance(image2, DataImage)
    # Check that the data are all the same: we have been playing only
    # with identity mappings
    yield np.testing.assert_array_equal, xyz_image2.get_data(), \
            image2.get_data()

