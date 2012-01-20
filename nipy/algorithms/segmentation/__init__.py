from .brain_segmentation import (initialize_parameters, brain_segmentation)
from .segmentation import Segmentation

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
