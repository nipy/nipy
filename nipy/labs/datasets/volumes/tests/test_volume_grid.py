# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Testing VolumeGrid interface.
"""

import nose
import copy

import numpy as np

# Local imports
from ..volume_grid import VolumeGrid
from ..volume_img import VolumeImg
from ...transforms.transform import Transform, CompositionError

def mapping(x, y, z):
    return 2*x, y, 0.5*z

def inverse_mapping(x, y, z):
    return 0.5*x, y, 2*z

def id(x, y, z):
    return x, y, z

################################################################################
# Tests
def test_constructor():
    yield np.testing.assert_raises, ValueError, VolumeGrid, None, \
        None, {}, 'e'


def test_volume_grid():
    """ Sanity testing of the VolumeGrid class.
    """
    transform = Transform('voxels', 'world', mapping)
    img = VolumeGrid(data=np.random.random((10, 10, 10)),
                    transform=transform,
                    )
    # Test that the repr doesn't raise an error
    yield repr, img

    # We cannot calculate the values in the world, because the transform 
    # is not invertible.
    
    yield np.testing.assert_raises, ValueError, \
                        img.values_in_world, 0, 0, 0
    yield np.testing.assert_raises, ValueError, \
        img.as_volume_img

    yield nose.tools.assert_equal, img, copy.copy(img)


def test_trivial_grid():
    """ Test resampling for an grid embedded in world space with an
        identity transform. 
    """
    N = 10
    identity = Transform('voxels', 'world', id, id)
    data = np.random.random((N, N, N))
    img = VolumeGrid(data=data,
                    transform=identity,
                    )
    x, y, z = np.random.random_integers(N, size=(3, 10)) - 1
    data_ = img.values_in_world(x, y, z)
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
    img1 = VolumeGrid(data=data,
                     transform=v2w_mapping,
                     )
    img2 = img1.composed_with_transform(identity)
    
    yield nose.tools.assert_equal, img2.world_space, 'world2'

    x, y, z = N*np.random.random(size=(3, 10))
    yield np.testing.assert_almost_equal, img1.values_in_world(x, y, z), \
        img2.values_in_world(x, y, z)

    yield nose.tools.assert_raises, CompositionError, \
            img1.composed_with_transform, identity.get_inverse()

    yield nose.tools.assert_raises, CompositionError, img1.resampled_to_img, \
            img2
    
    # Resample an image on itself: it shouldn't change much:
    img  = img1.resampled_to_img(img1)
    yield np.testing.assert_almost_equal, data, img.get_data()

    # Check that if I 'resampled_to_img' on an VolumeImg, I get an
    # VolumeImg, and vice versa 
    volume_image = VolumeImg(data, np.eye(4), 'world')
    identity  = Transform('voxels', 'world', id, id) 
    image = VolumeGrid(data, identity)
    image2 = image.resampled_to_img(volume_image)
    yield nose.tools.assert_true, isinstance(image2, VolumeImg)
    volume_image2 = volume_image.resampled_to_img(image)
    yield nose.tools.assert_true, isinstance(image2, VolumeGrid)
    # Check that the data are all the same: we have been playing only
    # with identity mappings
    yield np.testing.assert_array_equal, volume_image2.get_data(), \
            image2.get_data()


def test_as_volume_image():
    """ Test casting VolumeGrid to VolumeImg
    """
    N = 10
    v2w_mapping  = Transform('voxels', 'world2', id, id) 
    data = np.random.random((N, N, N))
    img1 = VolumeGrid(data=data,
                     transform=v2w_mapping,
                     )
    img2 = img1.as_volume_img()

    # Check that passing in the wrong shape raises an error
    yield nose.tools.assert_raises, ValueError, img1.as_volume_img, None, \
            (10, 10)

