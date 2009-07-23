"""
Package containing both generic configuration and testing stuff as well as
general purpose functions that are useful to a broader community and not
restricted to the neuroimaging community. This package may contain
third-party software included here for convenience.
"""

from onetime import OneTimeProperty, setattr_on_read
from tmpdirs import TemporaryDirectory, InTemporaryDirectory

from data import template_repo, data_repo

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
