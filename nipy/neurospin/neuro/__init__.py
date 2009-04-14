from image_classes import *
from affine_registration import affine_registration
from linear_model import linear_model
from statistical_test import cluster_stats, onesample_test
import fmri

try:
    from numpy.testing import Tester
except ImportError:
    from fff2.utils.nosetester import NoseTester as Tester
test = Tester().test
bench = Tester().bench


