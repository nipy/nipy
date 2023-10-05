""" Testing nibcompat module
"""

import nibabel as nib
import numpy as np
from numpy.testing import assert_array_equal

from ..nibcompat import get_affine, get_dataobj, get_header, get_unscaled_data


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
        assert get_header(img) == 1
        assert_array_equal(get_affine(img), np.eye(4))
        assert get_dataobj(img) == 3


def test_unscaled_data(in_tmp_path):
    shape = (2, 3, 4)
    data = np.random.normal(size=shape)
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
        inter = (0. if 'scl_inter' not in header else header['scl_inter'])
        if np.isnan(slope):
            slope, inter = dao.slope, dao.inter
        data_back = np.array(dao)
        assert np.allclose(data, data_back, atol=slope / 2.)
        header_copy = header.copy()
        header_copy['vox_offset'] = default_offset
        with open(fname, 'rb') as fobj:
            raw_back = header_copy.raw_data_from_fileobj(fobj)
        unscaled = get_unscaled_data(img_back)
        assert_array_equal(unscaled, raw_back)
        assert np.allclose(unscaled * slope + inter, data_back)
        # delete objects to allow file deletion on Windows
        del raw_back, unscaled
