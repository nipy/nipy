# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nose.tools import assert_equal

from numpy.testing import assert_array_almost_equal

from nipy.io.imageformats import load
from nipy.testing import funcfile
from nipy.neurospin.registration import Image4d, resample4d


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
    im4d = Image4d(im.get_data(), im.get_affine(), tr=2., tr_slices=0.0)
    x = resample4d(im4d)
    assert_array_almost_equal(im4d.array, x)
