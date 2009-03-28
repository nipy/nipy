"""This test module is primarily to test nipy's use of pynifti.  Adding tests to
pynifti code for the low-level functionality.
"""
import warnings

import numpy as np

from neuroimaging.testing import *

from neuroimaging.io.pyniftiio import PyNiftiIO

def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")

def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


def test_affine_analyze():
    # Test method1 from nifti1.h, Analyze 7.5 mapping.
    # pynifti stores numpy array's in xyz order
    data = np.zeros((4,3,2))
    img = PyNiftiIO(data)
    pdim = [2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]
    img._nim.pixdim = pdim
    # flip our pixdims so it's in zyx order and create diagonal 3x3 transform
    diag = np.diag(pdim[:3])
    xform = np.zeros((4,4))
    xform[:3, :3] = diag
    xform[3, 3] = 1
    assert img._nim.header['sform_code'] == 0
    assert img._nim.header['qform_code'] == 0
    assert np.allclose(img.affine, xform)

def test_affine_qform():
    # Test method2 from nifti1.h, qform mapping.
    data = np.zeros((4,3,2))
    img = PyNiftiIO(data)
    # set qform_code through the header
    hdr = img._nim.header
    hdr['qform_code'] = 1
    img._nim.updateHeader(hdr)
    qform = np.array([[1.0, 0.0, 0.0, 10.0],
                      [0.0, 2.0, 0.0, 20.0],
                      [0.0, 0.0, 3.0, 30.0],
                      [0.0, 0.0, 0.0, 1.0]])
    img._nim.setQForm(qform)
    # zyx order transform
    xform = qform.copy()
    assert np.all(img.affine == xform)

def test_affine_sform():
    # Test method3 from nifti1.h, sform mapping.
    data = np.zeros((4,3,2))
    img = PyNiftiIO(data)
    # set sform_code through header
    hdr = img._nim.header
    hdr['sform_code'] = 4
    img._nim.updateHeader(hdr)
    # sform from mni152 from fsl
    mni_xform = np.array([[-2.0, 0.0, 0.0, 90.0],
                          [0.0, 2.0, 0.0, -126.0],
                          [0.0, 0.0, 2.0, -72.0],
                          [0.0, 0.0, 0.0, 1.0]])
    img._nim.setSForm(mni_xform)
    # Create a zyx ordered MNI transform
    xform = mni_xform.copy()
    #print mni_xform
    #print '*'*40
    #print xform
    assert np.all(img.affine == xform)

