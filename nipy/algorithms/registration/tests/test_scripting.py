#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nipy.testing import funcfile
import nipy.algorithms.registration as reg


def test_space_time_realign():
    trans = reg.space_time_realign(funcfile, 2.0)

if __name__=="__main__":
    test_space_time_realign()
