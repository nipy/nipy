import numpy as np

from nipy.testing import datapjoin, assert_true, assert_equal, assert_raises

from ..image_list import ImageList
from ..base_image import BaseImage
from nipy.io.api import load_image


def test_image_list():
    img_path = datapjoin("test_fmri.nii.gz")
    img = load_image(img_path)
    imglst = ImageList.from_image(img)
    
    # Test empty ImageList
    emplst = ImageList()
    yield assert_equal, len(emplst.list), 0

    # Test non-image construction
    a = np.arange(10)
    yield assert_raises, ValueError, ImageList, a

    yield assert_raises, ValueError, ImageList.from_image, img, None

    yield assert_equal, img.get_data().shape, (128, 128, 13, 120)

    # length of image list should match number of frames
    yield assert_equal, len(imglst.list), img.get_data().shape[3]

    # Slicing an ImageList should return an ImageList
    sublist = imglst[2:5]
    yield assert_true, isinstance(sublist, ImageList)
    # Except when we're indexing one element
    yield assert_true, isinstance(imglst[0], BaseImage)

    # Verify get_data 
    yield assert_true, isinstance(sublist.get_data(), np.ndarray)


    # Test __setitem__
    sublist[2] = sublist[0]
    yield assert_equal, sublist[0].get_data().mean(), \
        sublist[2].get_data().mean()

    # Test iterator
    for x in sublist:
        yield assert_true, isinstance(x, BaseImage)
        yield assert_equal, x.get_data().shape, (128, 128, 13)
