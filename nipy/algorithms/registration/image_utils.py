# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np


def get_affine(im):
    """
    Get the 4x4 matrix representing the (spatial) affine transform of
    an image whether the input is a nipy object or a nibabel
    object.

    This function will be useless when the nipy image class gets a
    get_affine method that always returns a 4x4 matrix. Currently, the
    affine of an AffineImage is not a 4x4 matrix for images with
    dimension > 3.

    Parameters
    ----------
    im : image
      Either a nipy image or a nibabel image.
    """
    if hasattr(im, 'affine'):
        a = im.affine
        b = np.concatenate((a[0:3, 0:3], a[0:3, -1].reshape((3, 1))), axis=1)
        b = np.concatenate((b, np.array([0, 0, 0, 1]).reshape((1, 4))))
        return b
    else:
        return im.get_affine()
