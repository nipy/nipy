import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.io.imageformats import load, save
from nipy.testing import funcfile
from nipy.neurospin.registration.fmri_realign4d import Image4d, _resample4d

"""
def test_registration():
    aff = register(anat_img, anat_img)
    print aff
"""
"""
Image4d(array, to_world, tr, tr_slices=None, start=0.0, 
slice_order='ascending', interleaved=False, slice_axis=2):

"""        

im = load(funcfile) 

"""
def test_to_time():
    im4d = Image4d(im.get_data(), im.get_affine(), tr=2., 
                   slice_order='ascending', interleaved=False)
    assert_equal(im4d.to_time(0,0), 0.) 
    assert_equal(im4d.to_time(0,1), im4d.tr)     
    assert_equal(im4d.to_time(0,2), 2*im4d.tr)
    assert_equal(im4d.to_time(1,0), im4d.tr_slices)
    assert_equal(im4d.to_time(im4d.nslices,0), im4d.nslices*im4d.tr_slices)
"""    

def test_from_time():
    im4d = Image4d(im.get_data(), im.get_affine(), tr=2., 
                   slice_order='ascending', interleaved=False)
    assert_equal(im4d.from_time(0,0), 0.) 
    assert_equal(im4d.from_time(0,im4d.tr), 1.) 
    assert_equal(im4d.from_time(1,im4d.tr_slices), 0.) 
                   

def test_slice_timing(): 
    im4d = Image4d(im.get_data(), im.get_affine(), tr=2., 
                   slice_order='ascending', interleaved=False)
    clone_im4d = _resample4d(im4d)
