from __future__ import with_statement
from os.path import exists
import numpy as np
from nibabel import Nifti1Image, save
from numpy.testing import assert_equal
from ...utils.simul_multisubject_fmri_dataset import surrogate_3d_dataset
from ..parcel_io import (mask_parcellation, fixed_parcellation,
                         parcellation_based_analysis)
from ..hierarchical_parcellation import hparcel
from ..discrete_domain import grid_domain_from_shape
from nibabel.tmpdirs import InTemporaryDirectory


def test_mask_parcel():
    """ Test that mask parcellation performs correctly
    """
    n_parcels = 20
    shape = (10, 10, 10)
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))
    wim = mask_parcellation(mask_image, n_parcels)
    assert_equal(np.unique(wim.get_data()), np.arange(n_parcels))


def test_mask_parcel_multi_subj():
    """ Test that mask parcellation performs correctly
    """
    rng = np.random.RandomState(0); 
    n_parcels = 20
    shape = (10, 10, 10)
    n_subjects = 5
    mask_images = []
    with InTemporaryDirectory():
        for subject in range(n_subjects):
            path = 'mask%s.nii' % subject
            save(Nifti1Image((rng.rand(*shape) > .1).astype('u8'),
                             np.eye(4)), path)
            mask_images.append(path)

        wim = mask_parcellation(mask_images, n_parcels)
        assert_equal(np.unique(wim.get_data()), np.arange(n_parcels))


def test_parcel_intra_from_3d_image():
    """Test that a parcellation is generated, starting from an input 3D image
    """
    # Generate an image
    shape = (10, 10, 10)
    n_parcel, nn, mu = 10, 6, 1.
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))
    with InTemporaryDirectory() as dir_context:
        surrogate_3d_dataset(mask=mask_image, out_image_file='image.nii')

        #run the algo 
        for method in ['ward', 'kmeans', 'gkm']:
            osp = fixed_parcellation(mask_image, ['image.nii'], n_parcel, nn,
                                     method, dir_context, mu)
            result = 'parcel_%s.nii' % method
            assert exists(result)
            assert_equal(osp.k, n_parcel)


def test_parcel_intra_from_3d_images_list():
    """Test that a parcellation is generated, starting from a list of 3D images
    """
    # Generate an image
    shape = (10, 10, 10)
    n_parcel, nn, mu = 10, 6, 1.
    method = 'ward'
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))

    with InTemporaryDirectory() as dir_context:
        data_image = ['image_%d.nii' % i for i in range(5)]
        for datim in data_image:
            surrogate_3d_dataset(mask=mask_image, out_image_file=datim)

        #run the algo
        osp = fixed_parcellation(mask_image, data_image, n_parcel, nn,
                                 method, dir_context, mu)
        assert exists('parcel_%s.nii' % method)
        assert_equal(osp.k, n_parcel)


def test_parcel_intra_from_4d_image():
    """Test that a parcellation is generated, starting from a 4D image
    """
    # Generate an image
    shape = (10, 10, 10)
    n_parcel, nn, mu = 10, 6, 1.
    method = 'ward'
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))
    with InTemporaryDirectory() as dir_context:
        surrogate_3d_dataset(n_subj=10, mask=mask_image, 
                             out_image_file='image.nii')    
        osp = fixed_parcellation(mask_image, ['image.nii'], n_parcel, nn,
                                 method, dir_context, mu)
        assert exists('parcel_%s.nii' % method)
        assert_equal(osp.k, n_parcel)

def test_parcel_based_analysis():
    # Generate an image
    shape = (7, 8, 4)
    n_subj = 5
    n_parcel, nn, mu = 10, 6, 1.

    with InTemporaryDirectory() as dir_context:
        data_image = ['image_%d.nii' % i for i in range(5)]
        for datim in data_image:
            surrogate_3d_dataset(shape=shape, out_image_file=datim)
        ldata = np.random.randn(n_subj, np.prod(shape), 1)
        domain = grid_domain_from_shape(shape)
        parcels = hparcel(domain, ldata, n_parcel, mu=3.0)
        prfx = parcellation_based_analysis(
            parcels, data_image, test_id='one_sample', rfx_path='prfx.nii',
            condition_id='', swd=dir_context)
        assert exists('prfx.nii')
        assert np.abs(prfx).max() < 15

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
