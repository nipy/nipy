
Generating simulated activation maps
=====================================

.. currentmodule:: nipy.labs.utils.simul_multisubject_fmri_dataset

The module :mod:`nipy.labs.utils.simul_multisubject_fmri_dataset`
contains a various functions to create simulated activation maps in two, three
and four dimensions.  A 2D example is :func:`surrogate_2d_dataset`.  The
functions can position various activations and add noise, both as background
noise and jitter in the activation positions and amplitude.

These functions can be useful to test methods.

Example
--------

.. plot:: labs/plots/surrogate_array.py
    :include-source:


Function documentation
-------------------------

.. autofunction:: surrogate_2d_dataset

.. autofunction:: surrogate_3d_dataset

.. autofunction:: surrogate_4d_dataset
