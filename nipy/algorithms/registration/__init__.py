# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .resample import *
from .histogram_registration import *
from .affine import *
from .groupwise_registration import *

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench

