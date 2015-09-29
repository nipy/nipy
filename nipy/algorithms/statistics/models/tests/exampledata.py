from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.bin")
data = np.fromfile(filename, "<f8")
data.shape = (126,15)

y = data[:,0]
x = data[:,1:]
