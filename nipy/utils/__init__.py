# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
General utilities for code support.

These are modules that we (broadly-speaking) wrote; packages that other people
wrote, that we ship, go in the nipy.externals tree.
"""
from __future__ import absolute_import

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

from .utilities import is_iterable, is_numlike, seq_prod

from nipy.testing import Tester
test = Tester().test
bench = Tester().bench


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    Python does not show any DeprecationWarning by default.  Sometimes we do
    want to show a deprecation warning, when the deprecation is urgent, or the
    usage is probably a bug.
    """


class _NoValue:
    """Special keyword value.

    This class may be used as the default value assigned to a deprecated
    keyword in order to check if it has been given a user defined value.
    """
