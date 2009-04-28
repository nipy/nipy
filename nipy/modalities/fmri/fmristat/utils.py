"""
Utilities for fmristat
"""

__docformat__ = 'restructuredtext'

import gc, os, fpformat

from numpy import asarray

class WholeBrainNormalize(object):
    """
    This class constructs a callable object that
    is sometimes used to normalize fMRI data
    before applying a GLM.

    The normalization consists of dividing each point in the time series
    by the corresponding frame average of the fMRI image.
    """
    
    def __init__(self, fmri_image, mask=None):
        """
        :Parameters:
            `fmri_image` : A sequence of fMRI Images (can be a 4D Image)
            `mask` : A mask over which to average each fMRI volume.
        """
        if mask is not None:
            mask = np.asarray(mask)
            nvox = mask.astype(np.int32).sum()
        else:
            nvox = np.product(fmri_image[0].shape[1:])

        self.n = fmri_image.coordmap.shape[0]
        self.avg = np.zeros((self.n,))

        for i, d in data_generator(fmri_image):
            if mask is not None:
                d = d * mask # can't do in place as the slice points into a 
                             # memmap which may not be writable.
            self.avg[i] = d.sum() / nvox

    def __call__(self, data):
        """
        :Parameters:
            `fmri_data` : fMRI data assumed to have the same number
                          of frames as fmri_data when constructed
        :Returns: 
            `normalized` : normalized fMRI data: each frame is
                           divided by its masked average
        """
        fmri_data = np.asarray(fmri_data)
        out = np.zeros(fmri_data.shape)
        for i in range(self.n):
            out[i] = fmri_data[i] * 100. / self.avg[i]
        return out




