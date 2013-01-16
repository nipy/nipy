#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
This example illustrates  the impact of using a mixed-effects model
for the detection of the effects, when the first-level variance is known:
If the first level variance is very variable across observations, then taking
it into account gives more relibale detections, as seen in an ROC curve.

Requires matplotlib.

Author: Bertrand Thirion, 2012
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from nipy.algorithms.statistics.mixed_effects_stat import (
    generate_data, one_sample_ttest, t_stat)

# generate the data
N, P = 15, 500
V1 = np.random.randn(N, P) ** 2
effects = 0.5 * (np.random.randn(P) > 0)
Y = generate_data(np.ones(N), effects, .25, V1)

# compute the statistics
T1 = one_sample_ttest(Y, V1, n_iter=5)
T1 = [T1[effects == x] for x  in np.unique(effects)]
T2 = [t_stat(Y)[effects == x] for x  in np.unique(effects)]

# Derive ROC curves
ROC1 = np.array([np.sum(T1[1] > - x) for x  in np.sort(- T1[0])])\
    * 1. / T1[1].size
ROC2 = np.array([np.sum(T2[1] > - x) for x  in np.sort(- T2[0])])\
    * 1. / T1[1].size

# make a figure
FIG = plt.figure(figsize=(10, 5))
AX = FIG.add_subplot(121)
AX.plot(np.linspace(0, 1, len(ROC1)), ROC1, label='mixed effects')
AX.plot(np.linspace(0, 1, len(ROC2)), ROC2, label='t test')
AX.set_xlabel('false positives')
AX.set_ylabel('true positives')
AX.set_title('ROC curves for the detection of effects', fontsize=12)
AX.legend(loc='lower right')
AX = FIG.add_subplot(122)
AX.boxplot(T1, positions=[-0.1, .9])
AX.boxplot(T2, positions=[0.1, 1.1])
AX.set_xticks([0, 1])
AX.set_xlabel('simulated effects')
AX.set_ylabel('decision statistic')
AX.set_title('left: mixed effects model, \n right: standard t test',
             fontsize=12)
plt.show()
