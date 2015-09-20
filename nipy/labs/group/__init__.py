# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import absolute_import

from warnings import warn

from . import onesample
from . import twosample
from . import glm_twolevel
from . import permutation_test

warn('Module nipy.labs.group deprecated, will be removed',
     FutureWarning,
     stacklevel=2)

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
