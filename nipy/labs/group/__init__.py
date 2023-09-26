# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from warnings import warn

from . import glm_twolevel, onesample, permutation_test, twosample

warn('Module nipy.labs.group deprecated, will be removed',
     FutureWarning,
     stacklevel=2)
