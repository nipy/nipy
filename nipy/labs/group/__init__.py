# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from . import onesample
from . import twosample
from . import glm_twolevel
from . import permutation_test

from warnings import warn

warn('This module (nipy.labs.group) is deprecated and will be removed '
     'from future versions of nipy')

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
