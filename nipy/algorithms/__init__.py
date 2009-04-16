"""
Package containing generic algorithms such as registration, statistics,
simulation, etc.
"""
__docformat__ = 'restructuredtext'

import statistics
import fwhm, interpolation, kernel_smooth, regression

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
