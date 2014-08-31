""" Testing nibcompat module
"""
import numpy as np

import nibabel as nib

from nibabel.tmpdirs import InTemporaryDirectory

from ..nibcompat import (get_dataobj, get_affine, get_header,
                         get_unscaled_data)

from numpy.testing import assert_array_equal

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_funcs():
    class OldNib:
        def get_header(self):
            return 1
        def get_affine(self):
            return np.eye(4)
        _data = 3
    class NewNib:
        header = 1
        affine = np.eye(4)
        dataobj = 3
    for img in OldNib(), NewNib():
        assert_equal(get_header(img), 1)
        assert_array_equal(get_affine(img), np.eye(4))
        assert_equal(get_dataobj(img), 3)


def test_unscaled_data():
    shape = (2, 3, 4)
    data = np.random.normal(size=shape)
    with InTemporaryDirectory():
        for img_class, ext, default_offset in (
            # (nib.Spm99AnalyzeImage, '.img', 0),
            # (nib.Spm2AnalyzeImage, '.img', 0),
            (nib.Nifti1Pair, '.img', 0),
            (nib.Nifti1Image, '.nii', 352),
        ):
            img = img_class(data, np.eye(4))
            img.set_data_dtype(np.dtype(np.int16))
            fname = 'test' + ext
            nib.save(img, fname)
            img_back = nib.load(fname)
            header = get_header(img_back)
            dao = get_dataobj(img_back)
            slope = header['scl_slope']
            inter = (0. if not 'scl_inter' in header else header['scl_inter'])
            if np.isnan(slope):
                slope, inter = dao.slope, dao.inter
            data_back = np.array(dao)
            assert_true(np.allclose(data, data_back, atol=slope / 2.))
            header_copy = header.copy()
            header_copy['vox_offset'] = default_offset
            with open(fname, 'rb') as fobj:
                raw_back = header_copy.raw_data_from_fileobj(fobj)
            unscaled = get_unscaled_data(img_back)
            assert_array_equal(unscaled, raw_back)
            assert_true(np.allclose(unscaled * slope + inter, data_back))
            # delete objects to allow file deletion on Windows
            del raw_back, unscaled
