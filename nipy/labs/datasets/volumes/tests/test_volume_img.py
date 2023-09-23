# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the VolumeImg object.
"""

import copy

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from ...transforms.affine_transform import AffineTransform
from ...transforms.affine_utils import from_matrix_vector
from ...transforms.transform import Transform
from ..volume_img import CompositionError, VolumeImg


################################################################################
# Helper function
def rotation(theta, phi):
    """ Returns a rotation 3x3 matrix.
    """
    cos = np.cos
    sin = np.sin
    a1 = np.array([[cos(theta), -sin(theta), 0],
                [sin(theta),  cos(theta), 0],
                [         0,           0, 1]])
    a2 = np.array([[ 1,        0,         0],
                [ 0, cos(phi), -sin(phi)],
                [ 0, sin(phi),  cos(phi)]])
    return np.dot(a1, a2)

def id(x, y, z):
    return x, y, z


################################################################################
# Tests
def test_constructor():
    pytest.raises(AttributeError, VolumeImg, None,
                         None, 'foo')
    pytest.raises(ValueError, VolumeImg, None,
                         np.eye(4), 'foo', {}, 'e')


def test_identity_resample():
    """ Test resampling of the VolumeImg with an identity affine.
    """
    shape = (3, 2, 5, 2)
    data = np.random.randint(0, 10, shape)
    affine = np.eye(4)
    affine[:3, -1] = 0.5 * np.array(shape[:3])
    ref_im = VolumeImg(data, affine, 'mine')
    rot_im = ref_im.as_volume_img(affine, interpolation='nearest')
    assert_almost_equal(data, rot_im.get_fdata())
    # Now test when specifying only a 3x3 affine
    #rot_im = ref_im.as_volume_img(affine[:3, :3], interpolation='nearest')
    assert_almost_equal(data, rot_im.get_fdata())
    reordered_im = rot_im.xyz_ordered()
    assert_almost_equal(data, reordered_im.get_fdata())


def test_downsample():
    """ Test resampling of the VolumeImg with a 1/2 down-sampling affine.
    """
    shape = (6, 3, 6, 2)
    data = np.random.random(shape)
    affine = np.eye(4)
    ref_im = VolumeImg(data, affine, 'mine')
    rot_im = ref_im.as_volume_img(2*affine, interpolation='nearest')
    downsampled = data[::2, ::2, ::2, ...]
    x, y, z = downsampled.shape[:3]
    np.testing.assert_almost_equal(downsampled,
                                   rot_im.get_fdata()[:x, :y, :z, ...])


def test_resampling_with_affine():
    """ Test resampling with a given rotation part of the affine.
    """
    prng = np.random.RandomState(10)
    data = prng.randint(4, size=(1, 4, 4))
    img = VolumeImg(data, np.eye(4), 'mine', interpolation='nearest')
    for angle in (0, np.pi, np.pi/2, np.pi/4, np.pi/3):
        rot = rotation(0, angle)
        rot_im = img.as_volume_img(affine=rot)
        assert_almost_equal(np.max(data), np.max(rot_im.get_fdata()))


def test_reordering():
    """ Test the xyz_ordered method of the VolumeImg.
    """
    # We need to test on a square array, as rotation does not change
    # shape, whereas reordering does.
    shape = (5, 5, 5, 2, 2)
    data = np.random.random(shape)
    affine = np.eye(4)
    affine[:3, -1] = 0.5*np.array(shape[:3])
    ref_im = VolumeImg(data, affine, 'mine')
    # Test with purely positive matrices and compare to a rotation
    for theta, phi in np.random.randint(4, size=(5, 2)):
        rot = rotation(theta*np.pi/2, phi*np.pi/2)
        rot[np.abs(rot)<0.001] = 0
        rot[rot>0.9] = 1
        rot[rot<-0.9] = 1
        b = 0.5*np.array(shape[:3])
        new_affine = from_matrix_vector(rot, b)
        rot_im = ref_im.as_volume_img(affine=new_affine)
        assert_array_equal(rot_im.affine, new_affine)
        assert_array_equal(rot_im.get_fdata().shape, shape)
        reordered_im = rot_im.xyz_ordered()
        assert_array_equal(reordered_im.affine[:3, :3], np.eye(3))
        assert_almost_equal(reordered_im.get_fdata(), data)

    # Check that we cannot swap axes for non spatial axis:
    pytest.raises(ValueError, ref_im._swapaxes, 4, 5)

    # Create a non-diagonal affine, and check that we raise a sensible
    # exception
    affine[1, 0] = 0.1
    ref_im = VolumeImg(data, affine, 'mine')
    pytest.raises(CompositionError, ref_im.xyz_ordered)


    # Test flipping an axis
    data = np.random.random(shape)
    for i in (0, 1, 2):
        # Make a diagonal affine with a negative axis, and check that
        # can be reordered, also vary the shape
        shape = (i+1, i+2, 3-i)
        affine = np.eye(4)
        affine[i, i] *= -1
        img = VolumeImg(data, affine, 'mine')
        orig_img = copy.copy(img)
        x, y, z = img.get_world_coords()
        sample = img.values_in_world(x, y, z)
        img2 = img.xyz_ordered()
        # Check that img has not been changed
        assert img == orig_img
        x_, y_, z_ = img.get_world_coords()
        assert_array_equal(np.unique(x), np.unique(x_))
        assert_array_equal(np.unique(y), np.unique(y_))
        assert_array_equal(np.unique(z), np.unique(z_))
        sample2 = img.values_in_world(x, y, z)
        assert_array_equal(sample, sample2)


def test_eq():
    """ Test copy and equality for VolumeImgs.
    """
    import copy
    shape = (4, 3, 5, 2)
    data = np.random.random(shape)
    affine = np.random.random((4, 4))
    ref_im = VolumeImg(data, affine, 'mine')
    assert ref_im == ref_im
    assert ref_im == copy.copy(ref_im)
    assert ref_im == copy.deepcopy(ref_im)
    # Check that as_volume_img with no arguments returns the same image
    assert ref_im == ref_im.as_volume_img()
    copy_im = copy.copy(ref_im)
    copy_im.get_fdata()[0, 0, 0] *= -1
    assert ref_im != copy_im
    copy_im = copy.copy(ref_im)
    copy_im.affine[0, 0] *= -1
    assert ref_im != copy_im
    copy_im = copy.copy(ref_im)
    copy_im.world_space = 'other'
    assert ref_im != copy_im
    # Test repr
    assert isinstance(repr(ref_im), str)
    # Test init: should raise exception is not passing in right affine
    pytest.raises(Exception, VolumeImg, data,
                          np.eye(3, 3), 'mine')


def test_values_in_world():
    """ Test the evaluation of the data in world coordinate.
    """
    shape = (3, 5, 4, 2)
    data = np.random.random(shape)
    affine = np.eye(4)
    ref_im = VolumeImg(data, affine, 'mine')
    x, y, z = np.indices(ref_im.get_fdata().shape[:3])
    values = ref_im.values_in_world(x, y, z)
    np.testing.assert_almost_equal(values, data)


def test_resampled_to_img():
    """ Trivial test of resampled_to_img.
    """
    shape = (5, 4, 3, 2)
    data = np.random.random(shape)
    affine = np.random.random((4, 4))
    ref_im = VolumeImg(data, affine, 'mine')
    assert_almost_equal(
        data,
        ref_im.as_volume_img(affine=ref_im.affine).get_fdata())
    assert_almost_equal(
        data,
        ref_im.resampled_to_img(ref_im).get_fdata())

    # Check that we cannot resample to another image in a different
    # world.
    other_im = VolumeImg(data, affine, 'other')
    pytest.raises(CompositionError,
                         other_im.resampled_to_img, ref_im)

    # Also check that trying to resample on a non 3D grid will raise an
    # error
    pytest.raises(ValueError,
                         ref_im.as_volume_img, None, (2, 2))


def test_transformation():
    """ Test transforming images.
    """
    N = 10
    identity1  = Transform('world1', 'world2', id, id)
    identity2  = AffineTransform('world1', 'world2', np.eye(4))
    for identity in (identity1, identity2):
        data = np.random.random((N, N, N))
        img1 = VolumeImg(data=data,
                           affine=np.eye(4),
                           world_space='world1',
                           )
        img2 = img1.composed_with_transform(identity)

        assert img2.world_space == 'world2'

        x, y, z = N*np.random.random(size=(3, 10))
        assert_almost_equal(img1.values_in_world(x, y, z),
                            img2.values_in_world(x, y, z))

        pytest.raises(CompositionError,
                             img1.composed_with_transform,
                             identity.get_inverse())

        pytest.raises(CompositionError,
                             img1.resampled_to_img,
                             img2)

        # Resample an image on itself: it shouldn't change much:
        img  = img1.resampled_to_img(img1)
        assert_almost_equal(data, img.get_fdata())


def test_get_affine():
    shape = (1, 2, 3, 4)
    data = np.random.randint(0, 10, shape)
    affine = np.eye(4)
    ref_im = VolumeImg(data, affine, 'mine')
    np.testing.assert_equal(ref_im.affine, ref_im.get_affine())
