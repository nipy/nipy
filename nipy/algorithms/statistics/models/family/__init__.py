# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
'''
This module contains the one-parameter exponential families used
for fitting GLMs and GAMs.

These families are described in

   P. McCullagh and J. A. Nelder.  "Generalized linear models."
   Monographs on Statistics and Applied Probability.
   Chapman & Hall, London, 1983.

'''

from .family import (Gaussian, Family, Poisson, Gamma, InverseGaussian,
                     Binomial)
