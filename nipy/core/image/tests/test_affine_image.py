"""
The base image interface.
"""

import nose
import numpy as np

from ...transforms.affine_utils import from_matrix_vector
from ..xyz_image import XYZImage, CompositionError

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


################################################################################
# Tests

def test_identity_resample():
    """ Test resampling of the XYZImage with an identity affine.
    """
    shape = (5., 5., 5.)
    data = np.random.randint(0, 10, shape)
    affine = np.eye(4)
    affine[:3, -1] = 0.5*np.array(shape)
    ref_im = XYZImage(data, affine, 'mine')
    rot_im = ref_im.resampled_to_grid(affine, interpolation_order=0)
    yield np.testing.assert_almost_equal, data, rot_im.get_data()
    reordered_im = rot_im.xyz_ordered()
    yield np.testing.assert_almost_equal, data, reordered_im.get_data()


def test_downsample():
    """ Test resampling of the XYZImage with a 1/2 down-sampling affine.
    """
    shape = (6., 6., 6.)
    data = np.random.randint(0, 10, shape)
    affine = np.eye(4)
    ref_im = XYZImage(data, affine, 'mine')
    rot_im = ref_im.resampled_to_grid(2*affine, interpolation_order=0)
    downsampled = data[::2, ::2, ::2]
    x, y, z = downsampled.shape
    np.testing.assert_almost_equal(downsampled, 
                                   rot_im.get_data()[:x, :y, :z])


def test_reordering():
    """ Test the xyz_ordered method of the XYZImage.
    """
    shape = (5., 5., 5.)
    data = np.random.random(shape)
    affine = np.eye(4)
    affine[:3, -1] = 0.5*np.array(shape)
    ref_im = XYZImage(data, affine, 'mine')
    for theta, phi in np.random.randint(4, size=(10, 2)):
        rot = rotation(theta*np.pi/2, phi*np.pi/2)
        rot[np.abs(rot)<0.001] = 0
        rot[rot>0.9] = 1
        rot[rot<-0.9] = 1
        b = 0.5*np.array(shape)
        new_affine = from_matrix_vector(rot, b)
        rot_im = ref_im.resampled_to_grid(new_affine=new_affine)
        reordered_im = rot_im.xyz_ordered()
        yield np.testing.assert_array_equal, reordered_im.affine[:3, :3], \
                                    np.eye(3)
        yield np.testing.assert_almost_equal, reordered_im.get_data(), \
                                    data

    # Create a non-diagonal affine, and check that we raise a sensible
    # exception
    affine[1, 0] = 0.1
    ref_im = XYZImage(data, affine, 'mine')
    yield nose.tools.assert_raises, CompositionError, ref_im.xyz_ordered


def test_eq():
    """ Test copy and equality for XYZImages.
    """
    import copy
    shape = (5., 5., 5.)
    data = np.random.random(shape)
    affine = np.random.random((4, 4))
    ref_im = XYZImage(data, affine, 'mine')
    yield nose.tools.assert_equal, ref_im, ref_im
    yield nose.tools.assert_equal, ref_im, copy.copy(ref_im)
    yield nose.tools.assert_equal, ref_im, copy.deepcopy(ref_im)
    copy_im = copy.copy(ref_im)
    copy_im.get_data()[0, 0, 0] *= -1
    yield nose.tools.assert_not_equal, ref_im, copy_im
    copy_im = copy.copy(ref_im)
    copy_im.affine[0, 0] *= -1
    yield nose.tools.assert_not_equal, ref_im, copy_im
    copy_im = copy.copy(ref_im)
    copy_im.world_space = 'other'
    yield nose.tools.assert_not_equal, ref_im, copy_im
    # Test repr
    yield np.testing.assert_, isinstance(repr(ref_im), str)
    # Test init: should raise exception is not passing in right affine
    yield nose.tools.assert_raises, Exception, XYZImage, data, \
                np.eye(3, 3), 'mine'

def test_values_in_world():
    """ Test the evaluation of the data in world coordinate.
    """
    shape = (5., 5., 5.)
    data = np.random.random(shape)
    affine = np.eye(4)
    ref_im = XYZImage(data, affine, 'mine')
    x, y, z = np.indices(ref_im.get_data().shape)
    values = ref_im.values_in_world(x, y, z)
    np.testing.assert_almost_equal(values, data)

# XXX: We probably need a helper function 'images_almost_equal'

def test_resampled_to_img():
    """ Trivial test of resampled_to_img.
    """
    shape = (5., 5., 5.)
    data = np.random.random(shape)
    affine = np.random.random((4, 4))
    ref_im = XYZImage(data, affine, 'mine')
    yield np.testing.assert_almost_equal, data, \
                        ref_im.resampled_to_grid(ref_im.affine).get_data()
    yield np.testing.assert_almost_equal, data, \
                        ref_im.resampled_to_img(ref_im).get_data()
    other_im = XYZImage(data, affine, 'other')
    yield nose.tools.assert_raises, CompositionError, \
            other_im.resampled_to_img, ref_im


