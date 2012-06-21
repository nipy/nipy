# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This modules lauches a one-sample test on a dataset
Statistical significance is obtained using cluster-level inference
and permutation testing.

Author: Alexis Roche, Bertrand Thirion 2009-2012
"""
import numpy as np
from nibabel import Nifti1Image as Image

import nipy.labs.statistical_mapping as sm
from nipy.utils import example_data


def remake_images():
    # Get group data
    group_data = example_data.get_filename(
        'neurospin', 'language_babies', 'offset_002.npz')
    f = np.load(group_data)
    data, vardata, xyz = f['mat'], f['var'], f['xyz']
    dX = xyz[0].max() + 1
    dY = xyz[1].max() + 1
    dZ = xyz[2].max() + 1
    aux = np.zeros([dX, dY, dZ])
    data_images = []
    vardata_images = []
    mask_images = []
    for i in range(data.shape[0]):
        aux[list(xyz)] = data[i]
        data_images.append(Image(aux.copy(), np.eye(4)))
        aux[list(xyz)] = vardata[i]
        vardata_images.append(Image(aux.copy(), np.eye(4)))
        aux[list(xyz)] = 1
        mask_images.append(aux)

    return data_images, vardata_images, mask_images

data_images, vardata_images, mask_images = remake_images()

zimg, mask, nulls = sm.onesample_test(data_images,
                                      None,
                                      mask_images,
                                      'wilcoxon',
                                      permutations=1024,
                                      cluster_forming_th=0.01)
clusters, info = sm.cluster_stats(zimg, mask, 0.01, nulls=nulls)
