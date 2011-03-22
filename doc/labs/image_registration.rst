
Image registration
==================

.. currentmodule:: nipy.labs.image_registration

The module :mod:`nipy.labs.image_registration` currently
implements general 3D affine intensity-based image registration and
space-time realignment for fMRI data (simultaneous slice timing and
motion correction):

.. autosummary::
    :toctree: generated
 
    affine_register
    affine_resample
    image4d
    realign4d
    resample4d

Those functions take as inputs and return image objects as implemented 
in :mod:`nipy.io.imageformats`. 
