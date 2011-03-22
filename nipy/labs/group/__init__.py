# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import onesample
import twosample
import glm_twolevel
import permutation_test
import spatial_relaxation_onesample

from numpy.testing import Tester

test = Tester().test
bench = Tester().bench
