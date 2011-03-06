from numpy.testing import assert_almost_equal
from nipy.neurospin.spatial_models.parcel_io import *
from nipy.neurospin.utils.simul_multisubject_fmri_dataset import \
    surrogate_3d_dataset, surrogate_4d_dataset 
from tempfile import mkdtemp
from os.path import join, exists

def test_mask_parcel():
    """
    Test that mask parcellation performs correctly
    """
    nb_parcel = 20
    shape = (10, 10, 10)
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))
    wim = mask_parcellation(mask_image, nb_parcel)
    assert (np.unique(wim.get_data())==np.arange(nb_parcel)).all()
    
def test_mask_parcel_multi_subj():
    """
    Test that mask parcellation performs correctly
    """
    tempdir = mkdtemp()
    nb_parcel = 20
    shape = (10, 10, 10)
    nb_subj = 5
    mask_image = []
    for s in range(nb_subj):
        path = join(tempdir, 'mask%s.nii' %s)
        save(Nifti1Image((np.random.rand(*shape)>.1).astype('u8'),
                         np.eye(4)), path)
        mask_image.append(path)

    wim = mask_parcellation(mask_image, nb_parcel)
    assert (np.unique(wim.get_data())==np.arange(nb_parcel)).all()
    
def test_parcel_intra_from_3d_image():
    """
    test that a parcellation is generated, starting from an input 3D image
    """
    # Generate an image 
    tempdir = mkdtemp()
    shape = (10, 10, 10)
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))
    data_image = join(tempdir, 'image.nii')
    surrogate_3d_dataset(mask=mask_image, out_image_file=data_image)
    
    #run the algo
    n_parcel = 10
    nn = 6
    mu = 1.
    for method in ['ward', 'kmeans', 'gkm']:
        osp = fixed_parcellation(mask_image, [data_image], n_parcel, nn, 
                                    method, tempdir, mu)
        result = join(tempdir, 'parcel_%s.nii'%method)
        assert exists(result)
        assert osp.k==n_parcel

def test_parcel_intra_from_3d_images_list():
    """
    test that a parcellation is generated, starting from a list of 3D images
    """
    # Generate an image 
    tempdir = mkdtemp()
    shape = (10, 10, 10)
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))
    data_image = [join(tempdir, 'image_%d.nii'%i) for i in range(5)]
    for datim in data_image:
        surrogate_3d_dataset(mask=mask_image, out_image_file=datim)
    
    #run the algo
    n_parcel = 10
    nn = 6
    mu = 1.
    method = 'ward'
    osp = fixed_parcellation(mask_image, data_image, n_parcel, nn, 
                                method, tempdir, mu)
    result = join(tempdir, 'parcel_%s.nii'%method)
    assert exists(result)
    assert osp.k==n_parcel
    
def test_parcel_intra_from_4d_image():
    """
    test that a parcellation is generated, starting from a 4D image
    """
    # Generate an image 
    tempdir = mkdtemp()
    shape = (10, 10, 10)
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))
    data_image = join(tempdir, 'image.nii')
    surrogate_3d_dataset(nbsubj=10, mask=mask_image, out_image_file=data_image)
    
    #run the algo
    n_parcel = 10
    nn = 6
    mu = 1.
    method = 'ward'
    osp = fixed_parcellation(mask_image, [data_image], n_parcel, nn, 
                                method, tempdir, mu)
    result = join(tempdir, 'parcel_%s.nii'%method)
    assert exists(result)
    assert osp.k==n_parcel
    

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

