# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Package containing both generic configuration and testing stuff as well as
general purpose functions that are useful to a broader community and not
restricted to the neuroimaging community. This package may contain
third-party software included here for convenience.
"""

from onetime import OneTimeProperty, setattr_on_read
from tmpdirs import TemporaryDirectory, InTemporaryDirectory

from nibabel.data import make_datasource, DataError, datasource_or_bomber

# Module level datasource instances for convenience
from ..info import DATA_PKGS
templates = datasource_or_bomber(DATA_PKGS['nipy-templates'])
example_data = datasource_or_bomber(DATA_PKGS['nipy-data'])

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
