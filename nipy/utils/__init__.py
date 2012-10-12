# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
General utilities for code support.

These are modules that we (broadly-speaking) wrote; packages that other people
wrote, that we ship, go in the nipy.externals tree.
"""

from nibabel.data import make_datasource, DataError, datasource_or_bomber

# Module level datasource instances for convenience
from ..info import DATA_PKGS
templates = datasource_or_bomber(DATA_PKGS['nipy-templates'])
example_data = datasource_or_bomber(DATA_PKGS['nipy-data'])

try:
    example_data.get_filename()
except DataError:
    HAVE_EXAMPLE_DATA = False
else:
    HAVE_EXAMPLE_DATA = True

try:
    templates.get_filename()
except DataError:
    HAVE_TEMPLATES = False
else:
    HAVE_TEMPLATES = True

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench
