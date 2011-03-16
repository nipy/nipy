
Generating simulated activation maps
=====================================

.. currentmodule:: nipy.labs.utils.simul_2d_multisubject_fmri_dataset

The module :mod:`nipy.labs.utils.simul_2d_multisubject_fmri_dataset`
contains a function to create simulated 2D activation maps:
:func:`make_surrogate_array`. The function can position various
activations and add noise, both as background noise and jitter in the
activation positions and amplitude.

This function is useful to test methods.

Example
--------

.. plot:: labs/plots/surrogate_array.py
    :include-source:


Function documentation
-------------------------

.. autofunction:: make_surrogate_array

