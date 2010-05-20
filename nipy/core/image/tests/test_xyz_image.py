# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import copy

from nipy.core.image import xyz_image
from nipy.core.api import Image
import numpy as np
from nipy.core.reference.coordinate_map import compose, AffineTransform, CoordinateSystem
from nipy.algorithms.resample import resample

from nose.tools import assert_raises, assert_true, assert_false, assert_equal
from numpy.testing import assert_almost_equal, assert_array_equal
from nipy.testing import parametric


lps = xyz_image.lps_output_coordnames
ras = xyz_image.ras_output_coordnames

def generate_im():
    data = np.random.standard_normal((30,40,50))
    affine = np.array([[0,0,4,3],
                       [0,3.4,0,1],
                       [1.3,0,0,2],
                       [0,0,0,1]])
    affine_mapping = AffineTransform(CoordinateSystem('ijk'),
                                     CoordinateSystem(lps),
                                     affine)
    xyz_im = xyz_image.XYZImage(data, affine, 'ijk',
                                metadata={'abc':np.random.standard_normal(4)})
    im = Image(data, affine_mapping)
    ras_im = xyz_image.XYZImage(data, np.dot(np.diag([-1,-1,1,1]),affine), 'ijk',
                                metadata={'abc':np.random.standard_normal(4)},
                                lps=False)


    return im, xyz_im, ras_im 


@parametric
def test_xyz_names():
    im, _, _ = generate_im()
    im = im.renamed_reference(**{'x+LR':'x'})
    yield assert_raises(ValueError, xyz_image.XYZImage.from_image, im)


def test__eq__():
    _, xyz_im, ras_im = generate_im()
    new_xyz_im = xyz_image.XYZImage(xyz_im.get_data(), 
                                    xyz_im.affine.copy(),
                                    'ijk',
                                    metadata=xyz_im.metadata)

    yield assert_true, xyz_im == xyz_im
    yield assert_true, xyz_im == new_xyz_im
    yield assert_true, xyz_im == copy.copy(xyz_im)
    yield assert_true, xyz_im == copy.deepcopy(xyz_im)


def test_affine_shape():

    data = np.random.standard_normal((30,40,50))
    affine = np.random.standard_normal((5,4))
    yield assert_raises, ValueError, xyz_image.XYZImage, data, affine, 'ijk'

def test_reordered_renamed_etc():

    _, xyz_im, ras_im = generate_im()
    yield assert_raises, NotImplementedError, xyz_im.xyz_transform.reordered_range, ()
    yield assert_raises, NotImplementedError, xyz_im.xyz_transform.renamed_range, ()
    yield assert_raises, NotImplementedError, xyz_im.renamed_reference, ()
    yield assert_raises, NotImplementedError, xyz_im.reordered_reference, ()

    data = np.random.standard_normal((11,9,4))
    im = xyz_image.XYZImage(data, np.diag([3,4,5,1]), 'ijk')
    im_renamed = im.renamed_axes(i='slice')
    yield assert_equal, im_renamed.axes, \
        CoordinateSystem(coord_names=('slice', 'j', 'k'), 
                         name='voxel')


    data = np.random.standard_normal((3,4,5))
    affine = np.random.standard_normal((4,4))
    affine[-1] = [0,0,0,1]
    yield assert_raises, ValueError, xyz_image.XYZImage, data, affine, 'ij'

    data = np.random.standard_normal((3,4,5,6,7))
    xyz_im = xyz_image.XYZImage(data, affine, 'ijklm')
    yield assert_raises, ValueError, xyz_im.resampled_to_img, xyz_im


@parametric
def test_reordered_axes():
    _, xyz_im, ras = generate_im()

    xyz_reordered = xyz_im.reordered_axes([2,0,1])
    yield assert_array_equal(np.array(xyz_reordered), 
                             np.transpose(np.array(xyz_im), [2,0,1]))

    xyz_reordered = xyz_im.reordered_axes('kij')
    yield assert_array_equal(np.array(xyz_reordered), 
                             np.transpose(np.array(xyz_im), [2,0,1]))

    xyz_reordered = xyz_im.reordered_axes()
    yield assert_array_equal(np.array(xyz_reordered), 
                             np.transpose(np.array(xyz_im), [2,1,0]))

    yield assert_equal, xyz_im.metadata, xyz_reordered.metadata

def test_xyz_reference_axes():

    im, xyz_im, ras_im = generate_im()

    yield assert_equal, CoordinateSystem(lps, name='world'), xyz_im.reference
    yield assert_equal, CoordinateSystem('ijk', name='voxel'), xyz_im.axes

    yield assert_equal, CoordinateSystem(ras, name='world'), ras_im.reference
    yield assert_equal, CoordinateSystem('ijk', name='voxel'), ras_im.axes

def test_xyz_image():

    im, lps_im, ras_im = generate_im()

    for xyz_im, coords in zip([lps_im, ras_im],
                              [lps, ras]):
        xyz_cmap = xyz_im.xyz_transform

        yield assert_true,  xyz_cmap.function_domain.coord_names == ('i','j','k')
        yield assert_equal,  xyz_im.axes.coord_names, ('i','j','k')
        yield assert_equal, xyz_im.axes, xyz_cmap.function_domain

        yield assert_true,  xyz_cmap.function_range.coord_names == coords
        yield assert_equal, xyz_im.reference, xyz_cmap.function_range

        # test to_image from_image

        b = xyz_im.to_image()
        c = xyz_image.XYZImage.from_image(b)

        yield assert_equal, b.metadata, c.metadata
        yield assert_equal, xyz_im.metadata, c.metadata

        yield assert_almost_equal, b.coordmap.affine, c.coordmap.affine
        yield assert_almost_equal, xyz_im.coordmap.affine, c.coordmap.affine

        yield assert_equal, id(b._data), id(c._data)
        yield assert_almost_equal, id(xyz_im._data), id(c._data)


        b = xyz_im.xyz_ordered()
        b_cmap = b.xyz_transform

        #

        yield assert_equal,  b_cmap.function_domain.coord_names, ('k','j','i')
        yield assert_equal, b.axes.coord_names, ('k', 'j', 'i')


        yield assert_equal,  b_cmap.function_range.coord_names, coords

        yield assert_equal,  xyz_im.shape, im.shape
        yield assert_equal,  b.shape, im.shape[::-1]

        

def test_resample():
    im, xyz_im, ras = generate_im()

    xyz_im_resampled = xyz_im.resampled_to_affine(xyz_im.xyz_transform)
    yield assert_almost_equal, np.array(xyz_im_resampled), np.array(xyz_im)
    yield assert_equal, xyz_im.metadata, xyz_im_resampled.metadata

    xyz_im_resampled2 = xyz_im.resampled_to_img(xyz_im)
    yield assert_almost_equal, np.array(xyz_im_resampled2), np.array(xyz_im)
    yield assert_equal, xyz_im.metadata, xyz_im_resampled2.metadata
    # first call xyz_ordered

    xyz_im_xyz = xyz_im.xyz_ordered()
    xyz_im_xyz.xyz_ordered(positive=True)

    xyz_im_resampled = xyz_im_xyz.resampled_to_affine(xyz_im_xyz.xyz_transform)
    yield assert_almost_equal, np.array(xyz_im_resampled), np.array(xyz_im_xyz)

    xyz_im_resampled2 = xyz_im_xyz.resampled_to_img(xyz_im_xyz)
    yield assert_almost_equal, np.array(xyz_im_resampled2), np.array(xyz_im_xyz)

    # What we can't do is resample to
    # an array with axes ['k','j','i'], (i.e. transpose the data using resample_*) because we've assumed that
    # the axes (i.e. function_domain) are the same for these methods


def test_subsample():

    # This is how you would subsample with nipy.algorithms.resample
    # On the first axis, we'll take every 2nd,
    # on the second axis every 3rd, and on the 3rd every 4th

    im, xyz_im, ras = generate_im()

    subsample_matrix = np.array([[2,0,0,0],
                                 [0,3,0,0],
                                 [0,0,4,0],
                                 [0,0,0,1]])
                                
    subsampled_shape = np.array(xyz_im)[::2,::3,::4].shape
    subsample_coordmap = AffineTransform(xyz_im.xyz_transform.function_domain,
                                         xyz_im.xyz_transform.function_domain,
                                         subsample_matrix)
    target_coordmap = compose(xyz_im.xyz_transform, 
                              subsample_coordmap)

    # The images have the same output coordinates

    world_to_world_coordmap = AffineTransform(xyz_im.xyz_transform.function_range,
                                              xyz_im.xyz_transform.function_range,
                                              np.identity(4))

    im_subsampled = resample(xyz_im, target_coordmap,
                             world_to_world_coordmap,
                             shape=subsampled_shape)
    xyz_im_subsampled = xyz_image.XYZImage(np.array(im_subsampled),
                                           im_subsampled.affine,
                                           im_subsampled.coordmap.function_domain.coord_names)

    yield assert_almost_equal, np.array(xyz_im_subsampled), np.array(xyz_im)[::2,::3,::4]

    # We can now do subsampling with these methods.
    xyz_im_subsampled2 = xyz_im.resampled_to_affine(target_coordmap, 
                                                         shape=subsampled_shape)
    yield assert_almost_equal, np.array(xyz_im_subsampled2), np.array(xyz_im_subsampled)
    yield assert_true, xyz_im_subsampled2 == xyz_im_subsampled
    
def test_values_in_world():
    im, xyz_im, ras_im = generate_im()

    xyz_vals = xyz_im.xyz_transform(np.array([[3,4,5],
                                             [4,7,8]]))
    x = xyz_vals[:,0]
    y = xyz_vals[:,1]
    z = xyz_vals[:,2]

    v1, v2 = xyz_im.values_in_world(x,y,z)
    yield assert_almost_equal, v1, np.array(xyz_im)[3,4,5]
    yield assert_almost_equal, v2, np.array(xyz_im)[4,7,8]

    x2 = np.array([3,4,5])
    yield assert_raises, ValueError, xyz_im.values_in_world, x2,y,z

def test_xyz_ordered():
    data = np.random.standard_normal((30,40,50))
    affine = np.array([[2,0,4,3],
                       [0,3.4,0,1],
                       [1.3,0,0,2],
                       [0,0,0,1]])
    xyz_im = xyz_image.XYZImage(data, affine, 'ijk')
    yield assert_raises, ValueError, xyz_im.xyz_ordered

    affine = np.array([[-3,0,0,4],
                       [0,2,0,10],
                       [0,0,-4,13],
                       [0,0,0,1]])

    im = xyz_image.XYZImage(data, affine, 'ijk')
    im_re = im.reordered_axes('kji')
    im_re_xyz = im_re.xyz_ordered()
    yield assert_true, im_re_xyz == im

    im_re_xyz_positive = im_re.xyz_ordered(positive=True)


    # the xyz_ordered with positive=True
    # option is not as simple as just multiplying flipping
    # the signs of the affine

    # first, it will flip the data.
    # in this case, the 'i/x' and 'k/z' coordinates have negative
    # diagonal entries in the original affine
    # so the 'i' and the 'k' axis will be flipped

    yield assert_almost_equal, im_re_xyz_positive.get_data(), im.get_data()[::-1,:,::-1]

    # as for the affine, for the scalings on the diagonal
    # it does just change the signs,
    # but the last column changes, too

    T = np.dot(np.diag([-1,1,-1,1]), im.affine)
    yield assert_false, np.all(np.equal(im_re_xyz_positive.affine, T))
    yield assert_almost_equal, im_re_xyz_positive.affine[:3,:3], T[:3,:3]

    step_x, step_y, step_z = np.diag(T)[:3]
    n_x, n_y, n_z = im.shape
    start_x, start_y, start_z = affine[:3,-1]
    b = [start_x+(n_x-1)*(-step_x), start_y, start_z+(n_z-1)*(-step_z)]
    yield assert_almost_equal, b, im_re_xyz_positive.affine[:3,-1]

def test_flip():

    _, lps_im, ras_im = generate_im()

    print xyz_image.flip(lps_im).xyz_transform
    yield assert_almost_equal, xyz_image.flip(lps_im).get_data(), ras_im.get_data()
    yield assert_almost_equal, xyz_image.flip(ras_im).get_data(), lps_im.get_data()
    yield assert_equal, xyz_image.flip(ras_im).xyz_transform, lps_im.xyz_transform
    yield assert_equal, xyz_image.flip(ras_im), lps_im
    yield assert_equal, xyz_image.flip(lps_im), ras_im

def test_xyz_transform():

    T = xyz_image.XYZTransform(np.diag([3,4,5,1]), ['slice', 'frequency', 'phase'])
   
    yield assert_equal, T.function_domain.coord_names, ('slice', 'frequency', 'phase')
    yield assert_equal, T.function_range.coord_names, lps

    Tinv = T.inverse()

    # The inverse doesn't map a voxel to 'xyz', it maps
    # 'xyz' to a voxel, so it's not an XYZTransform

    yield assert_false, isinstance(Tinv, xyz_image.XYZTransform)
    yield assert_true, Tinv.__class__ == AffineTransform
    yield assert_almost_equal, Tinv.affine, np.diag([1/3., 1/4., 1/5., 1])
