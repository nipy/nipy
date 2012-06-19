from .brain_segmentation import (initialize_parameters, brain_segmentation)
from .vem import (gauss_dist, laplace_dist, vm_step_gauss, weighted_median,
                  vm_step_laplace, VEM)

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench

