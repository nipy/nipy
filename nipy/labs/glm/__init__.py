# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from warnings import warn

from .glm import contrast, load, models, ols

warn('Module nipy.labs.glm deprecated, will be removed. '
     'Please use nipy.modalities.fmri.glm instead.',
     FutureWarning,
     stacklevel=2)
