#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import numpy.testing as npt

from nibabel.tmpdirs import InTemporaryDirectory
import nibabel.eulerangles as euler

from nipy.testing import funcfile
import nipy.algorithms.registration as reg


def test_space_time_realign():
    with InTemporaryDirectory() as tmpdir:
        xform = reg.space_time_realign(funcfile, 2.0, out_name='./')


def test_aff2euler():
    xr = 0.1
    yr = -1.3
    zr = 3.1
    scales = (2.1, 3.2, 4.4)
    R = np.dot(euler.euler2mat(xr, yr, zr), np.diag(scales))
    aff = np.eye(4)
    aff[:3, :3] = R
    aff[:3, 3] = [11, 12, 13]
    npt.assert_almost_equal(reg.aff2euler(aff), (xr, yr, zr))
