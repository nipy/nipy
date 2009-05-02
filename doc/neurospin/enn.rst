
Empirical null
==============

.. currentmodule:: fff2.utils.emp_null

The :mod:`fff2.utils.emp_null` module contains a class that fits a
gaussian model to the central part of an histogram, following Schwartzman
et al, 2009. This is typically necessary to estimate a fdr when one is
not certain that the data behaves as a standard normal under H_0.

The `ENN` class learns its null distribution on the data provided at
initialisation. Two different methods can be used to set a threshold
from the null distribution: the :meth:`ENN.threshold` method returns the
threshold for a given false discovery rate, and thus accounts for
multiple comparisons with the given dataset; the
:meth:`ENN.uncorrected_threshold` returns the threshold for a given
uncorrected p-value, and as such does not account for multiple
comparisons.

Example
---------

If we use the empirical normal null estimator on a two gaussian mixture
distribution, with a central gaussian, and a wide one, it uses the
central distribution as a null hypothesis, and returns the threshold
followingr which the data can be claimed to belong to the wide gaussian:

.. plot:: neurospin/plots/enn_demo.py
    :include-source:

The threshold evaluated with the :meth:`ENN.threshold` method is around
2.8 (using the default p-value of 0.05). The
:meth:`ENN.uncorrected_threshold` return, for the same p-value, a
threshold of 1.9. It is necessary to use a higher p-value with
uncorrected comparisons.

Class documentation
--------------------


.. autoclass:: ENN
    :members:

    .. automethod:: __init__

____

**Reference**: Schwartzmann et al., NeuroImage 44 (2009) 71--82

