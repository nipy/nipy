# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing registration

"""
from nipy.neurospin.registration import *

from numpy.testing import assert_array_almost_equal

from nipy import load_image
from nipy.testing import anatfile


anat_img = load_image(anatfile)

def test_register():
    affine = register(anat_img, anat_img, subsampling=[4,4,4])
    assert_array_almost_equal(affine.as_affine(), np.eye(4))

def test_rigid_register():
    rigid = register(anat_img, anat_img, search='rigid', subsampling=[4,4,4])
    assert_array_almost_equal(rigid.as_affine(), np.eye(4))

def test_similarity_register():
    similarity = register(anat_img, anat_img, search='similarity', subsampling=[4,4,4])
    assert_array_almost_equal(similarity.as_affine(), np.eye(4))



        
