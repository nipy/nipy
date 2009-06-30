import affine_image, affine_imageII
import nipy.io.api as A
import numpy as np

def test_affine_image():

    # The file dummy.mnc was created with this commang 

    # rawtominc -input ../empty.raw dummy.mnc 30 40 50 -xstart 30 -xstep 3 -ystart 40 -ystep 4 -zstart 50 -zstep 5     

    im=A.load_image('/home/jtaylo/dummy.mnc')

    a = affine_image.AffineImage(np.array(im), im.affine, im.coordmap.input_coords.name)
    aII = affine_imageII.AffineImage(np.array(im), im.affine, im.coordmap.input_coords.coord_names)

    a_cmap = a.get_3dcoordmap()
    aII_cmap = aII.get_3dcoordmap()

    yield nose.assert_true,  a_cmap.input_coords.coord_names == ('axis0', 'axis1', 'axis2')
    yield nose.assert_true,  aII_cmap.input_coords.coord_names == ('i','j','k')

    yield nose.assert_true,  a_cmap.output_coords.coord_names == ('x','y','z')
    yield nose.assert_true,  aII_cmap.output_coords.coord_names == ('x','y','z')

    b=a.xyz_ordered()
    bII = aII.xyz_ordered()

    # The coordmap property of AffineImage could overwrite Image's
    # I haven't tried to do that yet.

    b_cmap = b.get_3dcoordmap()
    bII_cmap = bII.get_3dcoordmap()

    # I prefer the affine_imageII implementation
    # because you see that the axes have reversed order.
    # Just using a name for the coordinate system
    # loses this information

    yield nose.assert_true,  b_cmap.input_coords.coord_names == ('axis0', 'axis1', 'axis2')
    yield nose.assert_true,  bII_cmap.input_coords.coord_names == ('k','j','i')

    yield nose.assert_true,  b_cmap.output_coords.coord_names == ('x','y','z')
    yield nose.assert_true,  bII_cmap.output_coords.coord_names == ('x','y','z')

    np.testing.assert_almost_equal(b.affine, bII.affine)
    np.testing.assert_almost_equal(a.affine, aII.affine)

    yield nose.assert_true,  a.shape == im.shape
    yield nose.assert_true,  aII.shape == im.shape

    yield nose.assert_true,  b.shape == im.shape[::-1]
    yield nose.assert_true,  bII.shape == im.shape[::-1]


