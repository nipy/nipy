# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from nipy.core.image import affine_image
from nipy.core.image.affine_image import AffineImage
from nipy.core.api import Image
from nipy.core.reference.coordinate_map import compose, AffineTransform, CoordinateSystem
from nipy.algorithms.resample import resample

from nose.tools import (assert_true, assert_equal, assert_raises)

from numpy.testing import assert_array_equal, assert_almost_equal

def generate_im():
    data = np.random.standard_normal((30,40,50))
    affine = np.array([[0,0,4,3],
                       [0,3.4,0,1],
                       [1.3,0,0,2],
                       [0,0,0,1]])
    affine_mapping = AffineTransform(CoordinateSystem('ijk'),
                                     CoordinateSystem('xyz'),
                                     affine)
    affine_im = affine_image.AffineImage(data, affine, 'input')
    im = Image(data, affine_mapping)
    return im, affine_im


def test_affine_image():

    # The file dummy.mnc is available here:
    #
    # http://kff.stanford.edu/~jtaylo/affine_image_testfiles

    im, a = generate_im()

    a_cmap = a.spatial_coordmap

    yield assert_true,  a_cmap.function_domain.coord_names == ('axis0', 'axis1', 'axis2')
    yield assert_equal, a.coord_sys, 'input'

    yield assert_true,  a_cmap.function_range.coord_names == ('x','y','z')

    b=a.xyz_ordered()

    # The coordmap property of AffineImage could overwrite Image's
    # I haven't tried to do that yet.

    b_cmap = b.spatial_coordmap

    # I prefer the lpi_image implementation
    # because you see that the axes have reversed order.
    # Just using a name for the coordinate system
    # loses this information

    yield assert_true,  b_cmap.function_domain.coord_names == ('axis0', 'axis1', 'axis2')
    yield assert_equal, b.coord_sys, 'input' 

    yield assert_true,  b_cmap.function_range.coord_names == ('x','y','z')

    yield assert_true,  a.shape == im.shape

    yield assert_true,  b.shape == im.shape[::-1]

def test_resample():
    im, affine_im = generate_im()

    affine_im_resampled = affine_im.resampled_to_affine(affine_im.spatial_coordmap)
    yield assert_almost_equal, np.array(affine_im_resampled), np.array(affine_im)

    affine_im_resampled2 = affine_im.resampled_to_img(affine_im)
    yield assert_almost_equal, np.array(affine_im_resampled2), np.array(affine_im)

    # first call xyz_ordered

    affine_im_xyz = affine_im.xyz_ordered()
    affine_im_resampled = affine_im_xyz.resampled_to_affine(affine_im_xyz.spatial_coordmap)
    yield assert_almost_equal, np.array(affine_im_resampled), np.array(affine_im_xyz)

    affine_im_resampled2 = affine_im_xyz.resampled_to_img(affine_im_xyz)
    yield assert_almost_equal, np.array(affine_im_resampled2), np.array(affine_im_xyz)

def test_subsample():

    # This is how you would subsample with nipy.algorithms.resample
    # On the first axis, we'll take every 2nd,
    # on the second axis every 3rd, and on the 3rd every 4th

    im, affine_im = generate_im()
    subsample_matrix = np.array([[2,0,0,0],
                                 [0,3,0,0],
                                 [0,0,4,0],
                                 [0,0,0,1]])
                                
    subsampled_shape = np.array(affine_im)[::2,::3,::4].shape
    subsample_coordmap = AffineTransform(affine_im.spatial_coordmap.function_domain,
                                         affine_im.spatial_coordmap.function_domain,
                                         subsample_matrix)
    target_coordmap = compose(affine_im.spatial_coordmap, 
                              subsample_coordmap)

    # The images have the same output coordinates

    world_to_world_coordmap = AffineTransform(affine_im.spatial_coordmap.function_range,
                                              affine_im.spatial_coordmap.function_range,
                                              np.identity(4))
                                              

    im_subsampled = resample(affine_im, target_coordmap,
                             world_to_world_coordmap,
                             shape=subsampled_shape)
    affine_im_subsampled = affine_image.AffineImage(np.array(im_subsampled),
                                                    im_subsampled.affine,
                                                    im_subsampled.coordmap.function_domain.coord_names)

    yield assert_almost_equal, np.array(affine_im_subsampled), np.array(affine_im)[::2,::3,::4]

    # We can now do subsampling with these methods.
    affine_im_subsampled2 = affine_im.resampled_to_affine(target_coordmap, 
                                                         shape=subsampled_shape)
    yield assert_almost_equal, np.array(affine_im_subsampled2), np.array(affine_im_subsampled)
    
def test_values_in_world():
    im, affine_im = generate_im()

    xyz_vals = affine_im.spatial_coordmap(np.array([[3,4,5],
                                                [4,7,8]]))
    x = xyz_vals[:,0]
    y = xyz_vals[:,1]
    z = xyz_vals[:,2]

    v1, v2 = affine_im.values_in_world(x,y,z)
    yield assert_almost_equal, v1, np.array(affine_im)[3,4,5]
    yield assert_almost_equal, v2, np.array(affine_im)[4,7,8]


def test_4d_affine():
    arr = np.random.rand(3,4,5,6)
    assert_raises(ValueError, AffineImage, arr, np.eye(5), 'mni')
    im = AffineImage(arr, np.eye(4), 'mni')
    assert_equal(im.coordmap.ndims, (4,4))
    arr = np.random.rand(3,4,5,6,7)
    im = AffineImage(arr, np.eye(4), 'mni')
    assert_equal(im.coordmap.ndims, (5,5))

