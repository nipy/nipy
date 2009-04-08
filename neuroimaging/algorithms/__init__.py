"""
Package containing generic algorithms such as registration, statistics,
simulation, etc.
"""
__docformat__ = 'restructuredtext'

import statistics
import fwhm, interpolation, kernel_smooth, regression

from neuroimaging.testing import Tester
test = Tester().test
bench = Tester().bench
