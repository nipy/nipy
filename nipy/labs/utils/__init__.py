# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipy.testing import Tester

from .routines import (
    combinations,
    gamln,
    mahalanobis,
    median,
    permutations,
    psi,
    quantile,
    svd,
)
from .zscore import zscore

test = Tester().test
bench = Tester().bench
