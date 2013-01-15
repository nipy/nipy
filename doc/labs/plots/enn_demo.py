# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from nipy.algorithms.statistics.empirical_pvalue import NormalEmpiricalNull

x = np.c_[np.random.normal(size=1e4),
          np.random.normal(scale=4, size=1e4)]

enn = NormalEmpiricalNull(x)
enn.threshold(verbose=True)
