import onesample
import twosample
#import permutation_test
#import spatial_relaxation_onesample

try:
    from numpy.testing import Tester
except ImportError:
    from fff2.utils.nosetester import NoseTester as Tester
test = Tester().test
bench = Tester().bench
