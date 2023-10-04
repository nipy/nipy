# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test for the converters.
"""
import os
import tempfile

import nibabel as nib
import pytest

from .. import as_volume_img, save

data_file = os.path.join(nib.__path__[0], 'tests',
                                            'data', 'example4d.nii.gz')

def test_conversion():

    brifti_obj = nib.load(data_file)
    vol_img = as_volume_img(data_file)
    assert as_volume_img(vol_img) == vol_img
    assert as_volume_img(brifti_obj) == vol_img


def test_basics():
    pytest.raises(ValueError, as_volume_img, 'foobar')


def test_save():
    filename = tempfile.mktemp()
    try:
        img = as_volume_img(data_file)
        save(filename, img)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


try:
    import nifti
    def test_from_nifti():
        nim = nifti.NiftiImage(data_file)
        assert as_volume_img(data_file) == as_volume_img(nim)

except ImportError:
    pass
