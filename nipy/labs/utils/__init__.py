# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from routines import *
from zscore import zscore
from emp_null import ENN, FDR, three_classes_GMM_fit, Gamma_Gaussian_fit

from numpy.testing import Tester

test = Tester().test
bench = Tester().bench
