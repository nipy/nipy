
Empirical null
==============

.. currentmodule:: nipy.algorithms.statistics.empirical_pvalue

The :mod:`nipy.algorithms.statistics.empirical_pvalue` module contains a class
that fits a Gaussian model to the central part of an histogram, following
Schwartzman et al, 2009. This is typically necessary to estimate a FDR when one
is not certain that the data behaves as a standard normal under H_0.

The `NormalEmpiricalNull` class learns its null distribution on the data
provided at initialisation. Two different methods can be used to set a threshold
from the null distribution: the :meth:`NormalEmpiricalNull.threshold` method
returns the threshold for a given false discovery rate, and thus accounts for
multiple comparisons with the given dataset; the
:meth:`NormalEmpiricalNull.uncorrected_threshold` returns the threshold for a
given uncorrected p-value, and as such does not account for multiple
comparisons.

Example
-------

If we use the empirical normal null estimator on a two Gaussian mixture
distribution, with a central Gaussian, and a wide one, it uses the central
distribution as a null hypothesis, and returns the threshold following which the
data can be claimed to belong to the wide Gaussian:

.. plot:: labs/plots/enn_demo.py
    :include-source:

The threshold evaluated with the :meth:`NormalEmpiricalNull.threshold` method is
around 2.8 (using the default p-value of 0.05). The
:meth:`NormalEmpiricalNull.uncorrected_threshold` returns, for the same p-value,
a threshold of 1.9. It is necessary to use a higher p-value with uncorrected
comparisons.

Class documentation
-------------------

.. autoclass:: NormalEmpiricalNull
    :members:

    .. automethod:: __init__

____

**Reference**: Schwartzmann et al., NeuroImage 44 (2009) 71--82

