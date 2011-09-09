# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from ..histogram_registration import HistogramRegistration

import numpy as np 
from numpy.testing import assert_array_almost_equal

from .... import load_image
from ....testing import anatfile

anat_img = load_image(anatfile)

def _register(affine_type):
    R = HistogramRegistration(anat_img, anat_img)
    R.subsample([4,4,4])
    affine = R.optimize(affine_type)
    assert_array_almost_equal(affine.as_affine(), np.eye(4))

def test_rigid_register():
    _register('rigid')

def test_similarity_register():
    _register('similarity')

def test_affine_register():
    _register('affine')



        
