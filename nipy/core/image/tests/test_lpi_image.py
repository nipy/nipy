import copy
from nipy.core.image import lpi_image
from nipy.core.api import Image
import numpy as np
import nipy.testing as niptest
from nipy.core.reference.coordinate_map import compose, Affine as AffineTransform, CoordinateSystem
from nipy.algorithms.resample import resample

def generate_im():
    data = np.random.standard_normal((30,40,50))
    affine = np.array([[0,0,4,3],
                       [0,3.4,0,1],
                       [1.3,0,0,2],
                       [0,0,0,1]])
    affine_mapping = AffineTransform(affine,
                                     CoordinateSystem('ijk'),
                                     CoordinateSystem('xyz'))
    lpi_im = lpi_image.LPIImage(data, affine, 'ijk',
                                metadata={'abc':np.random.standard_normal(4)})
    im = Image(data, affine_mapping)
    return im, lpi_im

def test__eq__():
    _, lpi_im = generate_im()
    new_lpi_im = lpi_image.LPIImage(lpi_im.get_data(), 
                                    lpi_im.affine.copy(),
                                    'ijk',
                                    metadata=lpi_im.metadata)

    yield niptest.assert_true, lpi_im == lpi_im
    yield niptest.assert_true, lpi_im == new_lpi_im
    yield niptest.assert_true, lpi_im == copy.copy(lpi_im)
    yield niptest.assert_true, lpi_im == copy.deepcopy(lpi_im)


def test_affine_shape():

    data = np.random.standard_normal((30,40,50))
    affine = np.random.standard_normal((5,4))
    yield niptest.assert_raises, ValueError, lpi_image.LPIImage, data, affine, 'ijk'

def test_reordered_etc():

    _, lpi_im = generate_im()
    yield niptest.assert_raises, NotImplementedError, lpi_im.lpi_transform.reordered_output, ()
    yield niptest.assert_raises, NotImplementedError, lpi_im.lpi_transform.renamed_output, ()
    yield niptest.assert_raises, NotImplementedError, lpi_im.reordered_world, ()

    data = np.random.standard_normal((3,4,5))
    affine = np.random.standard_normal((4,4))
    yield niptest.assert_raises, ValueError, lpi_image.LPIImage, data, affine, 'ij'

    data = np.random.standard_normal((3,4,5,6,7))
    affine = np.random.standard_normal((4,4))
    lpi_im = lpi_image.LPIImage(data, affine, 'ijklm')
    yield niptest.assert_raises, ValueError, lpi_im.resampled_to_img, lpi_im

def test_reordered_axes():

    _, lpi_im = generate_im()

    lpi_reordered = lpi_im.reordered_axes([2,0,1])
    yield (niptest.assert_equal, np.array(lpi_reordered), 
           np.transpose(np.array(lpi_im), [2,0,1]))

    lpi_reordered = lpi_im.reordered_axes('kij')
    yield (niptest.assert_equal, np.array(lpi_reordered), 
           np.transpose(np.array(lpi_im), [2,0,1]))

    lpi_reordered = lpi_im.reordered_axes()
    yield (niptest.assert_equal, np.array(lpi_reordered), 
           np.transpose(np.array(lpi_im), [2,1,0]))

    yield niptest.assert_equal, lpi_im.metadata, lpi_reordered.metadata

def test_lpi_world_axes():

    im, lpi_im = generate_im()

    yield niptest.assert_equal, CoordinateSystem('xyz', name='world-LPI'), lpi_im.world
    yield niptest.assert_equal, CoordinateSystem('ijk', name='voxel'), lpi_im.axes

def test_lpi_image():

    im, lpi_im = generate_im()

    lpi_cmap = lpi_im.lpi_transform

    yield niptest.assert_true,  lpi_cmap.input_coords.coord_names == ('i','j','k')
    yield niptest.assert_equal,  lpi_im.axes.coord_names, ('i','j','k')

    yield niptest.assert_true,  lpi_cmap.output_coords.coord_names == ('x','y','z')

    b = lpi_im.xyz_ordered()
    b_cmap = b.lpi_transform


    yield niptest.assert_true,  b_cmap.input_coords.coord_names == ('k','j','i')
    yield niptest.assert_equal, b.axes.coord_names, ('k', 'j', 'i')

    yield niptest.assert_true,  b_cmap.output_coords.coord_names == ('x','y','z')

    yield niptest.assert_true,  lpi_im.shape == im.shape
    yield niptest.assert_true,  b.shape == im.shape[::-1]

def test_resample():
    im, lpi_im = generate_im()

    lpi_im_resampled = lpi_im.resampled_to_affine(lpi_im.lpi_transform)
    yield niptest.assert_almost_equal, np.array(lpi_im_resampled), np.array(lpi_im)
    yield niptest.assert_equal, lpi_im.metadata, lpi_im_resampled.metadata

    lpi_im_resampled2 = lpi_im.resampled_to_img(lpi_im)
    yield niptest.assert_almost_equal, np.array(lpi_im_resampled2), np.array(lpi_im)
    yield niptest.assert_equal, lpi_im.metadata, lpi_im_resampled2.metadata
    # first call xyz_ordered

    lpi_im_xyz = lpi_im.xyz_ordered()
    lpi_im_resampled = lpi_im_xyz.resampled_to_affine(lpi_im_xyz.lpi_transform)
    yield niptest.assert_almost_equal, np.array(lpi_im_resampled), np.array(lpi_im_xyz)

    lpi_im_resampled2 = lpi_im_xyz.resampled_to_img(lpi_im_xyz)
    yield niptest.assert_almost_equal, np.array(lpi_im_resampled2), np.array(lpi_im_xyz)

    # What we can't do is resample to
    # an array with axes ['k','j','i'], (i.e. transpose the data using resample_*) because we've assumed that
    # the axes (i.e. input_coords) are the same for these methods


def test_subsample():

    # This is how you would subsample with nipy.algorithms.resample
    # On the first axis, we'll take every 2nd,
    # on the second axis every 3rd, and on the 3rd every 4th

    im, lpi_im = generate_im()

    subsample_matrix = np.array([[2,0,0,0],
                                 [0,3,0,0],
                                 [0,0,4,0],
                                 [0,0,0,1]])
                                
    subsampled_shape = np.array(lpi_im)[::2,::3,::4].shape
    subsample_coordmap = AffineTransform(subsample_matrix, 
                                         lpi_im.lpi_transform.input_coords,
                                         lpi_im.lpi_transform.input_coords)
    target_coordmap = compose(lpi_im.lpi_transform, 
                              subsample_coordmap)

    # The images have the same output coordinates

    world_to_world_coordmap = AffineTransform(np.identity(4), 
                                              lpi_im.lpi_transform.output_coords,
                                              lpi_im.lpi_transform.output_coords)

    im_subsampled = resample(lpi_im, target_coordmap,
                             world_to_world_coordmap,
                             shape=subsampled_shape)
    lpi_im_subsampled = lpi_image.LPIImage(np.array(im_subsampled),
                                                      im_subsampled.affine,
                                                      im_subsampled.coordmap.input_coords.coord_names)

    yield niptest.assert_almost_equal, np.array(lpi_im_subsampled), np.array(lpi_im)[::2,::3,::4]

    # We can now do subsampling with these methods.
    lpi_im_subsampled2 = lpi_im.resampled_to_affine(target_coordmap, 
                                                         shape=subsampled_shape)
    yield niptest.assert_almost_equal, np.array(lpi_im_subsampled2), np.array(lpi_im_subsampled)
    yield niptest.assert_true, lpi_im_subsampled2 == lpi_im_subsampled
    
def test_values_in_world():
    im, lpi_im = generate_im()

    xyz_vals = lpi_im.lpi_transform(np.array([[3,4,5],
                                             [4,7,8]]))
    x = xyz_vals[:,0]
    y = xyz_vals[:,1]
    z = xyz_vals[:,2]

    v1, v2 = lpi_im.values_in_world(x,y,z)
    yield niptest.assert_almost_equal, v1, np.array(lpi_im)[3,4,5]
    yield niptest.assert_almost_equal, v2, np.array(lpi_im)[4,7,8]

    x2 = np.array([3,4,5])
    yield niptest.assert_raises, ValueError, lpi_im.values_in_world, x2,y,z

def test_xyz_ordered():
    data = np.random.standard_normal((30,40,50))
    affine = np.array([[2,0,4,3],
                       [0,3.4,0,1],
                       [1.3,0,0,2],
                       [0,0,0,1]])
    lpi_im = lpi_image.LPIImage(data, affine, 'ijk')
    yield niptest.assert_raises, ValueError, lpi_im.xyz_ordered

    
