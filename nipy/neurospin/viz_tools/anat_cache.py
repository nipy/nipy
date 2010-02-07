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

from nipy.io.imageformats import load

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
        from nipy.utils.data import templates

        if cls.anat is not None:
            return cls.anat, cls.anat_sform, cls.anat_max
        filename = templates.get_filename(
                            'ICBM152', '1mm', 'T1_brain.nii.gz')
        if not os.path.exists(filename):
            raise OSError('Cannot find template file T1_brain.nii.gz'
                    'required to plot anatomy. Possible path: %s'
                    % filename)
        anat_im = load(filename)
        anat = anat_im.get_data()
        anat = anat.astype(np.float)
        anat_mask = ndimage.morphology.binary_fill_holes(anat > 0)
        anat = np.ma.masked_array(anat, np.logical_not(anat_mask))
        cls.anat_sform = anat_im.get_affine()
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



