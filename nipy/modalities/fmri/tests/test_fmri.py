# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import gc

import numpy as np
import pytest

from nipy.core.api import AffineTransform as AfT
from nipy.core.api import Image, parcels
from nipy.io.api import load_image, save_image
from nipy.modalities.fmri.api import FmriImageList, axis0_generator
from nipy.testing import funcfile


@pytest.mark.filterwarnings("ignore:"
                            "Default `strict` currently False:"
                            "FutureWarning")
def test_write(in_tmp_path):
    fname = 'myfile.nii'
    img = load_image(funcfile)
    save_image(img, fname)
    test = FmriImageList.from_image(load_image(fname))
    assert test[0].affine.shape == (4,4)
    assert img[0].affine.shape == (5,4)
    # Check the affine...
    A = np.identity(4)
    A[:3,:3] = img[:,:,:,0].affine[:3,:3]
    A[:3,-1] = img[:,:,:,0].affine[:3,-1]
    assert np.allclose(test[0].affine, A)
    del test


def test_iter():
    img = load_image(funcfile)
    img_shape = img.shape
    exp_shape = (img_shape[0],) + img_shape[2:]
    j = 0
    for i, d in axis0_generator(img.get_fdata()):
        j += 1
        assert d.shape == exp_shape
        del(i); gc.collect()
    assert j == img_shape[1]


def test_subcoordmap():
    img = load_image(funcfile)
    subcoordmap = img[3].coordmap
    xform = img.affine[:,1:]
    assert np.allclose(subcoordmap.affine[1:], xform[1:])
    assert np.allclose(subcoordmap.affine[0], [0,0,0,img.coordmap([3,0,0,0])[0]])


def test_labels1():
    img = load_image(funcfile)
    data = img.get_fdata()
    parcelmap = Image(img[0].get_fdata(), AfT('kji', 'zyx', np.eye(4)))
    parcelmap = (parcelmap.get_fdata() * 100).astype(np.int32)
    v = 0
    for i, d in axis0_generator(data, parcels(parcelmap)):
        v += d.shape[1]
    assert v == parcelmap.size
