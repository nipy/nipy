import numpy as np

from nipy.testing import datapjoin, assert_true, assert_equal, assert_raises, \
    assert_almost_equal

from nipy.core.image.image_list import ImageList
from nipy.core.image.image import Image
from nipy.io.api import load_image

def test_image_list():
    img_path = datapjoin("test_fmri.nii.gz")
    img = load_image(img_path)
    imglst = ImageList.from_image(img, axis=-1)
    
    # Test empty ImageList
    emplst = ImageList()
    yield assert_equal, len(emplst.list), 0

    # Test non-image construction
    a = np.arange(10)
    yield assert_raises, ValueError, ImageList, a

    yield assert_raises, ValueError, ImageList.from_image, img, None

    # check all the axes

    for i in range(4):
        order = range(4)
        order.remove(i)
        order.insert(0,i)
        img_re_i = img.reordered_reference(order).reordered_axes(order)
        imglst_i = ImageList.from_image(img, axis=i)

        yield assert_equal, imglst_i.list[0].shape, img_re_i.shape[1:]
        
        # check the affine as well

        yield assert_almost_equal, imglst_i.list[0].affine, \
            img_re_i.affine[1:,1:]

    yield assert_equal, img.shape, (128, 128, 13, 120)

    # length of image list should match number of frames
    yield assert_equal, len(imglst.list), img.shape[3]

    # check the affine
    A = np.identity(4)
    A[:3,:3] = img.affine[:3,:3]
    A[:3,-1] = img.affine[:3,-1]
    yield assert_almost_equal, imglst.list[0].affine, A

    # Slicing an ImageList should return an ImageList
    sublist = imglst[2:5]
    yield assert_true, isinstance(sublist, ImageList)
    # Except when we're indexing one element
    yield assert_true, isinstance(imglst[0], Image)

    # Verify array interface
    # test __array__
    yield assert_true, isinstance(np.asarray(sublist), np.ndarray)

    # Test __setitem__
    sublist[2] = sublist[0]
    yield assert_equal, np.asarray(sublist[0]).mean(), \
        np.asarray(sublist[2]).mean()

    # Test iterator
    for x in sublist:
        yield assert_true, isinstance(x, Image)
        yield assert_equal, x.shape, (128, 128, 13)
