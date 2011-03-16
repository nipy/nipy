
Mask-extraction utilities
==========================

.. currentmodule:: nipy.labs.utils.mask

The module :mod:`nipy.labs.utils.mask` contains utilities to extract
brain masks from fMRI data:

.. autosummary::
    :toctree: generated
 
    compute_mask
    compute_mask_files
    compute_mask_sessions

The :func:`compute_mask_files` and :func:`compute_mask_sessions`
functions work with Nifti files rather than numpy ndarrays. This is
convenient to reduce memory pressure when working with long time series,
as there is no need to store the whole series in memory.

