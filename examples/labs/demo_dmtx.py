# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Examples of design matrices specification and and computation
(event-related design, FIR design, etc)

Author : Bertrand Thirion: 2009-2010
"""
print __doc__

import numpy as np
import pylab as mp
import nipy.labs.utils.design_matrix as dm

# frame times
tr = 1.0
nscans = 128
frametimes = np.linspace(0, (nscans - 1) * tr, nscans)

# experimental paradigm
conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c3', 'c3', 'c3']
onsets = [30, 70, 100, 10, 30, 90, 30, 40, 60]
hrf_model = 'Canonical'
motion = np.cumsum(np.random.randn(128, 6), 0)
add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']

#event-related design matrix
paradigm = dm.EventRelatedParadigm(conditions, onsets)

X1 = dm.DesignMatrix(
    frametimes, paradigm, drift_model='Polynomial', drift_order=3,
    add_regs=motion, add_reg_names=add_reg_names)

# block design matrix
duration = 7 * np.ones(9)
paradigm = dm.BlockParadigm(con_id=conditions, onset=onsets,
                             duration=duration)

X2 = dm.DesignMatrix(frametimes, paradigm, drift_model='Polynomial',
                         drift_order=3)

# FIR model
paradigm = dm.EventRelatedParadigm(conditions, onsets)
hrf_model = 'FIR'
X3 = dm.DesignMatrix(frametimes, paradigm, hrf_model='FIR',
                      drift_model='Polynomial', drift_order=3,
                      fir_delays=range(1, 6))

# plot the results
fig = mp.figure(figsize=(10, 6))
ax = mp.subplot(1, 3, 1)
X1.show(ax=ax)
ax.set_title('Event-related design matrix', fontsize=12)
ax = mp.subplot(1, 3, 2)
X2.show(ax=ax)
ax.set_title('Block design matrix', fontsize=12)
ax = mp.subplot(1, 3, 3)
X3.show(ax=ax)
ax.set_title('FIR design matrix', fontsize=12)
mp.subplots_adjust(top=0.9, bottom=0.25)
mp.show()
