
from os.path import exists

import numpy as np
from nibabel import Nifti1Image

from ...utils.simul_multisubject_fmri_dataset import surrogate_3d_dataset
from ..bsa_io import make_bsa_image


def test_parcel_intra_from_3d_images_list(in_tmp_path):
    """Test that a parcellation is generated, starting from a list of 3D images
    """
    # Generate an image
    shape = (5, 5, 5)
    contrast_id = 'plop'
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))
    #mask_images = [mask_image for _ in range(5)]

    data_image = ['image_%d.nii' % i for i in range(5)]
    for datim in data_image:
        surrogate_3d_dataset(mask=mask_image, out_image_file=datim)

    #run the algo
    landmark, hrois = make_bsa_image(
        mask_image, data_image, threshold=10., smin=0, sigma=1.,
        prevalence_threshold=0, prevalence_pval=0.5, write_dir=in_tmp_path,
        algorithm='density', contrast_id=contrast_id)

    assert landmark == None
    assert len(hrois) == 5
    assert exists(f'density_{contrast_id}.nii')
    assert exists(f'prevalence_{contrast_id}.nii')
    assert exists(f'AR_{contrast_id}.nii')
    assert exists(f'CR_{contrast_id}.nii')
