# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# models - Statistical Models
#

__docformat__ = 'restructuredtext'

from nipy.fixes.scipy.stats.models.info import __doc__

import nipy.fixes.scipy.stats.models.model
import nipy.fixes.scipy.stats.models.regression
import nipy.fixes.scipy.stats.models.glm

__all__ = filter(lambda s:not s.startswith('_'),dir())

from numpy.testing import Tester
test = Tester().test
