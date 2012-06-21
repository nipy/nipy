from .brain_segmentation import BrainT1Segmentation
from .segmentation import Segmentation

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
