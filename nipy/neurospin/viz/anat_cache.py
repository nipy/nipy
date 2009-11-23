"""
3D visualization of activation maps using Mayavi

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD

# Standard library imports
import os

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
from scipy import ndimage

from nifti import NiftiImage

# The sform for MNI templates
mni_sform = np.array([[-1, 0, 0,   90],
                      [ 0, 1, 0, -126],
                      [ 0, 0, 1,  -72],
                      [ 0, 0, 0,   1]])

mni_sform_inv = np.linalg.inv(mni_sform)


################################################################################
# Caching of the MNI template. 
################################################################################

class _AnatCache(object):
    """ Class to store the anat array in cache, to avoid reloading it
        each time.
    """
    anat        = None
    anat_sform  = None
    blurred     = None

    @classmethod
    def get_anat(cls):
        # XXX: still relying on fff2
        import fff2.data
        if cls.anat is not None:
            return cls.anat, cls.anat_sform, cls.anat_max
        anat_im = NiftiImage(
                    os.path.join(os.path.dirname(
                        os.path.realpath(fff2.data.__file__)),
                        'MNI152_T1_1mm_brain.nii.gz'
                    ))
        anat = anat_im.data.T
        anat = anat.astype(np.float)
        anat_mask = ndimage.morphology.binary_fill_holes(anat > 0)
        anat = np.ma.masked_array(anat, np.logical_not(anat_mask))
        cls.anat_sform = anat_im.sform
        cls.anat = anat
        cls.anat_max = anat.max()
        return cls.anat, cls.anat_sform, cls.anat_max

    @classmethod
    def get_blurred(cls):
        if cls.blurred is not None:
            return cls.blurred
        anat, _, _ = cls.get_anat()
        cls.blurred = ndimage.gaussian_filter(
                (ndimage.morphology.binary_fill_holes(
                    ndimage.gaussian_filter(
                            (anat > 4800).astype(np.float), 6)
                        > 0.5
                    )).astype(np.float),
                2).T.ravel()
        return cls.blurred



