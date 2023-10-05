# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import gc
from os.path import split as psplit

import nibabel.eulerangles as euler
import numpy as np
import numpy.testing as npt

import nipy.algorithms.registration as reg
from nipy.io.api import load_image, save_image
from nipy.testing import funcfile


def test_space_time_realign(in_tmp_path):
    path, fname = psplit(funcfile)
    original_affine = load_image(funcfile).affine
    path, fname = psplit(funcfile)
    froot, _ = fname.split('.', 1)
    # Make another image with .nii extension and extra dot in filename
    save_image(load_image(funcfile), 'my.test.nii')
    for in_fname, out_fname in ((funcfile, froot + '_mc.nii.gz'),
                                ('my.test.nii', 'my.test_mc.nii.gz')):
        xforms = reg.space_time_realign(in_fname, 2.0, out_name='.')
        assert np.allclose(xforms[0].as_affine(), np.eye(4), atol=1e-7)
        assert not np.allclose(xforms[-1].as_affine(), np.eye(4), atol=1e-3)
        img = load_image(out_fname)
        npt.assert_almost_equal(original_affine, img.affine)
    # To avoid Windows "file ... used by another process" error when
    # removing temporary directory.
    gc.collect()


def test_aff2euler():
    xr = 0.1
    yr = -1.3
    zr = 3.1
    scales = (2.1, 3.2, 4.4)
    R = np.dot(euler.euler2mat(xr, yr, zr), np.diag(scales))
    aff = np.eye(4)
    aff[:3, :3] = R
    aff[:3, 3] = [11, 12, 13]
    npt.assert_almost_equal(reg.aff2euler(aff), (xr, yr, zr))
