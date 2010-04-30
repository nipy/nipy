""" Testing registration

"""

import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.io.imageformats import load, save
from nipy.testing import anatfile, funcfile
from nipy.neurospin import image_registration as ireg


anat_img = load(anatfile)


def test_registration():
    aff = ireg.affine_register(anat_img,
                               anat_img)
    print aff


        
