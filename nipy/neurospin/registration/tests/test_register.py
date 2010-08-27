# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing registration

"""

import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy import load_image
from nipy.testing import anatfile, funcfile
from nipy.neurospin.registration import register


anat_img = load_image(anatfile)


def test_registration():
    aff = register(anat_img, anat_img)
    print aff


        
