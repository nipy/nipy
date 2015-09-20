from __future__ import absolute_import

from os.path import exists

import numpy as np

from nibabel import Nifti1Image

from ...utils.simul_multisubject_fmri_dataset import surrogate_3d_dataset
from ..bsa_io import make_bsa_image

from nose.tools import assert_true
from numpy.testing import assert_equal

from nibabel.tmpdirs import InTemporaryDirectory


def test_parcel_intra_from_3d_images_list():
    """Test that a parcellation is generated, starting from a list of 3D images
    """
    # Generate an image
    shape = (5, 5, 5)
    contrast_id = 'plop'
    mask_image = Nifti1Image(np.ones(shape), np.eye(4))
    #mask_images = [mask_image for _ in range(5)]

    with InTemporaryDirectory() as dir_context:
        data_image = ['image_%d.nii' % i for i in range(5)]
        for datim in data_image:
            surrogate_3d_dataset(mask=mask_image, out_image_file=datim)

        #run the algo
        landmark, hrois = make_bsa_image(
            mask_image, data_image, threshold=10., smin=0, sigma=1.,
            prevalence_threshold=0, prevalence_pval=0.5, write_dir=dir_context,
            algorithm='density', contrast_id=contrast_id)

        assert_equal(landmark, None)
        assert_equal(len(hrois), 5)
        assert_true(exists('density_%s.nii' % contrast_id))
        assert_true(exists('prevalence_%s.nii' % contrast_id))
        assert_true(exists('AR_%s.nii' % contrast_id))
        assert_true(exists('CR_%s.nii' % contrast_id))


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
