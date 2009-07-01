from nipy.core.image import affine_image, affine_imageII
import nipy.io.api as io
import numpy as np
import nipy.testing as niptest
from nipy.core.reference.coordinate_map import compose, Affine as AffineTransform
from nipy.algorithms.resample import resample

def test_affine_image():

    # The file dummy.mnc is available here:
    #
    # http://kff.stanford.edu/~jtaylo/affine_image_testfiles


    im=io.load_image('/home/jtaylo/dummy.mnc')

    a = affine_image.AffineImage(np.array(im), im.affine, im.coordmap.input_coords.name)
    aII = affine_imageII.AffineImage(np.array(im), im.affine, im.coordmap.input_coords.coord_names)

    a_cmap = a.spatial_coordmap
    aII_cmap = aII.spatial_coordmap

    yield niptest.assert_true,  a_cmap.input_coords.coord_names == ('axis0', 'axis1', 'axis2')
    yield niptest.assert_equal, a.coord_sys, 'input'
    yield niptest.assert_true,  aII_cmap.input_coords.coord_names == ('i','j','k')
    yield niptest.assert_equal,  aII.axis_names, ('i','j','k')

    yield niptest.assert_true,  a_cmap.output_coords.coord_names == ('x','y','z')
    yield niptest.assert_true,  aII_cmap.output_coords.coord_names == ('x','y','z')

    b=a.xyz_ordered()
    bII = aII.xyz_ordered()

    # The coordmap property of AffineImage could overwrite Image's
    # I haven't tried to do that yet.

    b_cmap = b.spatial_coordmap
    bII_cmap = bII.spatial_coordmap

    # I prefer the affine_imageII implementation
    # because you see that the axes have reversed order.
    # Just using a name for the coordinate system
    # loses this information

    yield niptest.assert_true,  b_cmap.input_coords.coord_names == ('axis0', 'axis1', 'axis2')
    yield niptest.assert_true,  bII_cmap.input_coords.coord_names == ('k','j','i')
    yield niptest.assert_equal, bII.axis_names, ('k', 'j', 'i')
    yield niptest.assert_equal, b.coord_sys, 'input-reordered' # XXX tagging on "reordered" is probably overkill

    yield niptest.assert_true,  b_cmap.output_coords.coord_names == ('x','y','z')
    yield niptest.assert_true,  bII_cmap.output_coords.coord_names == ('x','y','z')

    np.testing.assert_almost_equal(b.affine, bII.affine)
    np.testing.assert_almost_equal(a.affine, aII.affine)

    yield niptest.assert_true,  a.shape == im.shape
    yield niptest.assert_true,  aII.shape == im.shape

    yield niptest.assert_true,  b.shape == im.shape[::-1]
    yield niptest.assert_true,  bII.shape == im.shape[::-1]

def test_resample():
    im = io.load_image('/home/jtaylo/dummy.mnc')
    im._data = np.random.standard_normal(im._data.shape)

    affine_im = affine_imageII.AffineImage(np.array(im), im.affine, ['i','j','k'])

    affine_im_resampled = affine_im.resampled_to_affine(affine_im.spatial_coordmap)
    yield niptest.assert_almost_equal, np.array(affine_im_resampled), np.array(affine_im)

    affine_im_resampled2 = affine_im.resampled_to_img(affine_im)
    yield niptest.assert_almost_equal, np.array(affine_im_resampled2), np.array(affine_im)

    affine_im = affine_image.AffineImage(np.array(im), im.affine, 'voxel')
    affine_im_resampled = affine_im.resampled_to_affine(affine_im.spatial_coordmap)
    yield niptest.assert_almost_equal, np.array(affine_im_resampled), np.array(affine_im)

    affine_im_resampled2 = affine_im.resampled_to_img(affine_im)
    yield niptest.assert_almost_equal, np.array(affine_im_resampled2), np.array(affine_im)

    # first call xyz_ordered

    affine_im_xyz = affine_im.xyz_ordered()
    affine_im_resampled = affine_im_xyz.resampled_to_affine(affine_im_xyz.spatial_coordmap)
    yield niptest.assert_almost_equal, np.array(affine_im_resampled), np.array(affine_im_xyz)

    affine_im_resampled2 = affine_im_xyz.resampled_to_img(affine_im_xyz)
    yield niptest.assert_almost_equal, np.array(affine_im_resampled2), np.array(affine_im_xyz)

    affine_im_xyz = affine_image.AffineImage(np.array(im), im.affine, 'voxel')
    affine_im_resampled = affine_im_xyz.resampled_to_affine(affine_im_xyz.spatial_coordmap)
    yield niptest.assert_almost_equal, np.array(affine_im_resampled), np.array(affine_im_xyz)

    affine_im_resampled2 = affine_im_xyz.resampled_to_img(affine_im_xyz)
    yield niptest.assert_almost_equal, np.array(affine_im_resampled2), np.array(affine_im_xyz)

    # What we can't do is resample to
    # an array with axes ['k','j','i'], (i.e. transpose the data using resample_*) because we've assumed that
    # the axes (i.e. input_coords) are the same for these methods

    # It can be done with nipy.algorithms.resample, but not the resample_* methods of AffineImage

def test_subsample():
    # We can't even do subsampling with these methods because
    # in the proposal the axes of the affine are always assumed to be the same
    # as self in the resample_* methods.

    # This is how you would subsample with nipy.algorithms.resample
    # On the first axis, we'll take every 2nd,
    # on the second axis every 3rd, and on the 3rd every 4th

    im = io.load_image('/home/jtaylo/dummy.mnc')
    im._data = np.random.standard_normal(im._data.shape)

    affine_im = affine_imageII.AffineImage(np.array(im), im.affine, ['i','j','k'])
    subsample_matrix = np.array([[2,0,0,0],
                                 [0,3,0,0],
                                 [0,0,4,0],
                                 [0,0,0,1]])
                                
    subsampled_shape = affine_im[::2,::3,::4].shape
    subsample_coordmap = AffineTransform(subsample_matrix, affine_im.spatial_coordmap.input_coords,
                                         affine_im.spatial_coordmap.input_coords)
    target_coordmap = compose(affine_im.spatial_coordmap, 
                              subsample_coordmap)

    # The images have the same output coordinates

    world_to_world_coordmap = AffineTransform(np.identity(4), affine_im.spatial_coordmap.output_coords,
                                              affine_im.spatial_coordmap.output_coords)

    im_subsampled = resample(affine_im, target_coordmap,
                             world_to_world_coordmap,
                             shape=subsampled_shape)
    affine_im_subsampled = affine_imageII.AffineImage(np.array(im_subsampled),
                                                      im_subsampled.affine,
                                                      im_subsampled.coordmap.input_coords.coord_names)

    yield niptest.assert_almost_equal, np.array(affine_im_subsampled), np.array(affine_im)[::2,::3,::4]
    
def test_values_in_world():
    im = io.load_image('/home/jtaylo/dummy.mnc')
    im._data = np.random.standard_normal(im._data.shape)

    affine_im = affine_imageII.AffineImage(np.array(im), im.affine, ['i','j','k'])

    xyz_vals = affine_im.spatial_coordmap(np.array([[3,4,5],
                                                    [4,7,8]]))
    x = xyz_vals[:,0]
    y = xyz_vals[:,1]
    z = xyz_vals[:,2]

    v1, v2 = affine_im.values_in_world(x,y,z)
    yield niptest.assert_almost_equal, v1, np.array(affine_im)[3,4,5]
    yield niptest.assert_almost_equal, v2, np.array(affine_im)[4,7,8]


