from nipy.testing import Tester

from .brain_segmentation import BrainT1Segmentation
from .segmentation import Segmentation, moment_matching

test = Tester().test
bench = Tester().bench
