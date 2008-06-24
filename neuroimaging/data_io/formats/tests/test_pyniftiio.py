"""This test module is primarily to test nipy's use of pynifti.  Adding tests to
pynifti code for the low-level functionality.
"""

import numpy as np

from neuroimaging.externals.scipy.testing import *

from neuroimaging.data_io.formats.pyniftiio import PyNiftiIO

# Test File io
"""
open file read only 'r'
open readwrite 'r+'
create new file from array
create new file from an Image (copy data array and header)
NOTE:  Need to update to the modular branch of pynifti, there's a few bug fixes
    to saving image attr to a file for images created from arrays. 2008-06-21
"""

def test_affine_analyze():
    # Test method1 from nifti1.h, Analyze 7.5 mapping.
    # pynifti stores numpy array's in zyx order
    data = np.zeros((4,3,2))
    img = PyNiftiIO(data)
    assert img._img.header['sform_code'] == 0
    assert img._img.header['qform_code'] == 0
    pdim = [2.0, 3.0, 4.0, 1.0]
    qoff = [20.0, 30.0, 40.0]
    # Note: pixdim attr in pynifti images is nifti pixdim element's 1 through 7
    #     (does not include qfac, pixdim[0])
    img._img.pixdim = pdim
    img._img.qoffset = qoff
    # create an array to test against
    xform = np.array([[4.0, 0.0, 0.0, 44.0],
                     [0.0, 3.0, 0.0, 33.0],
                     [0.0, 0.0, 2.0, 22.0],
                     [0.0, 0.0, 0.0, 1.0]])
    assert np.allclose(img.affine, xform)

def test_affine_qform():
    # Test method2 from nifti1.h, qform mapping.
    data = np.zeros((4,3,2))
    img = PyNiftiIO(data)
    # set qform_code through the header
    hdr = img._img.header
    hdr['qform_code'] = 1
    img._img.updateHeader(hdr)
    qform = np.array([[1.0, 0.0, 0.0, 10.0],
                      [0.0, 2.0, 0.0, 20.0],
                      [0.0, 0.0, 3.0, 30.0],
                      [0.0, 0.0, 0.0, 1.0]])
    img._img.setQForm(qform)
    # zyx order transform
    xform = np.array([[3.0, 0.0, 0.0, 33.0],
                      [0.0, 2.0, 0.0, 22.0],
                      [0.0, 0.0, 1.0, 11.0],
                      [0.0, 0.0, 0.0, 1.0]])
    assert np.all(img.affine == xform)

def test_affine_sform():
    # Test method3 from nifti1.h, sform mapping.
    data = np.zeros((4,3,2))
    img = PyNiftiIO(data)
    # set sform_code through header
    hdr = img._img.header
    hdr['sform_code'] = 4 # mni coords
    img._img.updateHeader(hdr)
    mni_xform = np.array([[2.0, 0.0, 0.0, 90.0],
                          [0.0, 2.0, 0.0, -126.0],
                          [0.0, 0.0, 2.0, -72.0],
                          [0.0, 0.0, 0.0, 1.0]])
    img._img.setSForm(mni_xform)
    # Create a zyx ordered MNI transform
    xform = np.array([[2.0, 0.0, 0.0, -70.0],
                      [0.0, 2.0, 0.0, -124.0],
                      [0.0, 0.0, 2.0, 92.0],
                      [0.0, 0.0, 0.0, 1.0]])
    assert np.all(img.affine == xform)
