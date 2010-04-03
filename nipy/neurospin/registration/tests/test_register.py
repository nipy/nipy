""" Testing registration

"""

import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.io.imageformats import load, save
from nipy.testing import anatfile, funcfile
from nipy.neurospin.registration import register


anat_img = load(anatfile)


def test_registration():
    aff = register(anat_img, anat_img)
    print aff


        
