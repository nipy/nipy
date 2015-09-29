from __future__ import absolute_import
from .brain_segmentation import BrainT1Segmentation
from .segmentation import Segmentation, moment_matching

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
