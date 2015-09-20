# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test surrogate data generation.
"""
from __future__ import absolute_import

import numpy as np
from nose.tools import assert_true
from nibabel import Nifti1Image

from ..simul_multisubject_fmri_dataset import \
    surrogate_2d_dataset, surrogate_3d_dataset, surrogate_4d_dataset 


def test_surrogate_array():
    """ Check that with no noise, the surrogate activation correspond to
        the ones that we specify. 2D version
    """
    # We can't use random positions, as the positions have to be
    # far-enough not to overlap.
    pos   = np.array([[ 2, 10],
                      [10,  4],
                      [19, 15],
                      [15, 19],
                      [5, 18]])
    ampli = np.random.random(5)
    data = surrogate_2d_dataset(n_subj=1, noise_level=0, spatial_jitter=0,
                                signal_jitter=0, pos=pos, shape=(20,20),
                                ampli=ampli).squeeze()
    x, y = pos.T
    np.testing.assert_array_equal(data[x, y], ampli)


def test_surrogate_array_3d():
    """ Check that with no noise, the surrogate activation correspond to
        the ones that we specify. 3D version
    """
    # We can't use random positions, as the positions have to be
    # far-enough not to overlap.
    pos   = np.array([[ 2, 10, 2],
                      [10,  4, 4],
                      [18, 13, 18],
                      [13, 18, 5],
                      [5, 18, 18]])
    ampli = np.random.random(5)
    data = surrogate_3d_dataset(n_subj=1, noise_level=0, spatial_jitter=0,
                                signal_jitter=0, pos=pos, shape=(20,20,20),
                                ampli=ampli).squeeze()
    x, y, z = pos.T
    np.testing.assert_array_equal(data[x, y, z], ampli)


def test_surrogate_array_3d_write():
    """ Check that 3D version spits files when required
    """
    from os import path
    from tempfile import mkdtemp
    write_path = path.join(mkdtemp(), 'img.nii') 
    shape = (5, 6, 7)
    data = surrogate_3d_dataset(shape=shape, out_image_file=write_path)
    assert_true(path.isfile(write_path))

def test_surrogate_array_3d_mask():
    """ Check that 3D version works when a mask is provided
    """
    shape = (5, 6, 7)
    mask = np.random.rand(*shape) > 0.5
    mask_img = Nifti1Image(mask.astype(np.uint8), np.eye(4))
    img = surrogate_3d_dataset(mask=mask_img)
    mean_image  = img[mask].mean()
    assert_true((img[mask == 0] == 0).all())


def test_surrogate_array_4d_shape():
    """Run the 4D datageneration; check the output shape and length
    """
    shape = (5, 6, 7)
    out_shape = shape + (1,)
    imgs = surrogate_4d_dataset(shape)
    assert_true(not np.any(np.asarray(imgs[0].shape) - np.asarray(out_shape)))
    n_sess = 3
    imgs = surrogate_4d_dataset(shape, n_sess=n_sess)
    assert_true(imgs[0].shape == out_shape)
    assert_true(len(imgs) == n_sess)
    n_scans = 5
    out_shape = shape + (n_scans,)
    imgs = surrogate_4d_dataset(shape, n_scans=n_scans)
    assert_true(imgs[0].shape == (out_shape))


def test_surrogate_array_4d_write():
    """Run the 4D data generation; check that output images are written
    """
    from os import path
    from tempfile import mkdtemp
    n_sess = 3
    write_paths = [path.join(mkdtemp(), 'img_%d.nii' % i) 
                   for i in range(n_sess)]
    shape = (5, 6, 7)
    imgs = surrogate_4d_dataset(shape, out_image_file=write_paths[0])
    assert_true(path.isfile(write_paths[0]))
    imgs = surrogate_4d_dataset(shape, n_sess=n_sess, 
                                out_image_file=write_paths)
    for wp in write_paths:
        assert_true(path.isfile(wp))


def test_surrogate_array_4d_mask():
    """Run the 4D version, with masking
    """
    shape = (5, 5, 5)
    mask = np.random.rand(*shape) > 0.5
    mask_img = Nifti1Image(mask.astype(np.uint8), np.eye(4))
    imgs = surrogate_4d_dataset(mask=mask_img)
    mean_image  = imgs[0].get_data()[mask].mean()
    assert_true((imgs[0].get_data()[mask == 0] < mean_image / 2).all())


def test_surrogate_array_4d_dmtx():
    """Run the 4D version, with design_matrix provided
    """
    shape = (5, 5, 5)
    n_scans = 25
    out_shape = shape + (n_scans,)
    dmtx = np.random.randn(n_scans, 3)
    imgs = surrogate_4d_dataset(shape, dmtx=dmtx)
    assert_true(not np.any(np.asarray(imgs[0].shape) - np.asarray(out_shape)))


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
