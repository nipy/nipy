from os.path import exists

import numpy as np
from nibabel import Nifti1Image, save
from numpy.testing import assert_array_equal

from ...utils.simul_multisubject_fmri_dataset import surrogate_3d_dataset
from ..discrete_domain import grid_domain_from_shape
from ..hierarchical_parcellation import hparcel
from ..parcel_io import (
    fixed_parcellation,
    mask_parcellation,
    parcellation_based_analysis,
)


def test_mask_parcel():
    """ Test that mask parcellation performs correctly
    """
    n_parcels = 20
    shape = (10, 10, 10)
    mask_image = Nifti1Image(np.ones(shape).astype('u1'), np.eye(4))
    wim = mask_parcellation(mask_image, n_parcels)
    assert_array_equal(np.unique(wim.get_fdata()), np.arange(n_parcels))


def test_mask_parcel_multi_subj(in_tmp_path):
    """ Test that mask parcellation performs correctly
    """
    rng = np.random.RandomState(0);
    n_parcels = 20
    shape = (10, 10, 10)
    n_subjects = 5
    mask_images = []
    for subject in range(n_subjects):
        path = f'mask{subject}.nii'
        arr = rng.rand(*shape) > .1
        save(Nifti1Image(arr.astype('u1'), np.eye(4)), path)
        mask_images.append(path)

    wim = mask_parcellation(mask_images, n_parcels)
    assert_array_equal(np.unique(wim.get_fdata()), np.arange(n_parcels))


def test_parcel_intra_from_3d_image(in_tmp_path):
    """Test that a parcellation is generated, starting from an input 3D image
    """
    # Generate an image
    shape = (10, 10, 10)
    n_parcel, nn, mu = 10, 6, 1.
    mask_image = Nifti1Image(np.ones(shape).astype('u1'), np.eye(4))
    surrogate_3d_dataset(mask=mask_image, out_image_file='image.nii')

    #run the algo
    for method in ['ward', 'kmeans', 'gkm']:
        osp = fixed_parcellation(mask_image, ['image.nii'], n_parcel, nn,
                                 method, in_tmp_path, mu)
        result = f'parcel_{method}.nii'
        assert exists(result)
        assert osp.k == n_parcel


def test_parcel_intra_from_3d_images_list(in_tmp_path):
    """Test that a parcellation is generated, starting from a list of 3D images
    """
    # Generate an image
    shape = (10, 10, 10)
    n_parcel, nn, mu = 10, 6, 1.
    method = 'ward'
    mask_image = Nifti1Image(np.ones(shape).astype('u1'), np.eye(4))

    data_image = ['image_%d.nii' % i for i in range(5)]
    for datim in data_image:
        surrogate_3d_dataset(mask=mask_image, out_image_file=datim)

    #run the algo
    osp = fixed_parcellation(mask_image, data_image, n_parcel, nn,
                             method, in_tmp_path, mu)
    assert exists(f'parcel_{method}.nii')
    assert osp.k == n_parcel


def test_parcel_intra_from_4d_image(in_tmp_path):
    """Test that a parcellation is generated, starting from a 4D image
    """
    # Generate an image
    shape = (10, 10, 10)
    n_parcel, nn, mu = 10, 6, 1.
    method = 'ward'
    mask_image = Nifti1Image(np.ones(shape).astype('u1'), np.eye(4))
    surrogate_3d_dataset(n_subj=10, mask=mask_image,
                         out_image_file='image.nii')
    osp = fixed_parcellation(mask_image, ['image.nii'], n_parcel, nn,
                             method, in_tmp_path, mu)
    assert exists(f'parcel_{method}.nii')
    assert osp.k == n_parcel


def test_parcel_based_analysis(in_tmp_path):
    # Generate an image
    shape = (7, 8, 4)
    n_subj = 5
    n_parcel, nn, mu = 10, 6, 1.

    data_image = ['image_%d.nii' % i for i in range(5)]
    for datim in data_image:
        surrogate_3d_dataset(shape=shape, out_image_file=datim)
    ldata = np.random.randn(n_subj, np.prod(shape), 1)
    domain = grid_domain_from_shape(shape)
    parcels = hparcel(domain, ldata, n_parcel, mu=3.0)
    prfx = parcellation_based_analysis(
        parcels, data_image, test_id='one_sample', rfx_path='prfx.nii',
        condition_id='', swd=in_tmp_path)
    assert exists('prfx.nii')
    assert np.abs(prfx).max() < 15
