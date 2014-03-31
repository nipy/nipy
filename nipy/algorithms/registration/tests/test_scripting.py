#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nipy.testing import funcfile
import nipy.algorithms.registration as reg
from nibabel.tmpdirs import InTemporaryDirectory


def test_space_time_realign():
    with InTemporaryDirectory() as tmpdir:
        trans = reg.space_time_realign(funcfile, 2.0)

