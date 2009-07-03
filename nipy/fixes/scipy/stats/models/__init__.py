#
# models - Statistical Models
#

__docformat__ = 'restructuredtext'

from nipy.fixes.scipy.stats.models.info import __doc__

import nipy.fixes.scipy.stats.models.model
import nipy.fixes.scipy.stats.models.regression
from nipy.fixes.scipy.stats.models.glm import Model as glm

__all__ = filter(lambda s:not s.startswith('_'),dir())

from numpy.testing import Tester
test = Tester().test
