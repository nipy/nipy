# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
General utilities for code support.

These are modules that we (broadly-speaking) wrote; packages that other people
wrote, that we ship, go in the nipy.externals tree.
"""

import functools
import warnings

import numpy as np
from nibabel.data import DataError, datasource_or_bomber, make_datasource

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


# Numpy sctypes (np.sctypes removed in Numpy 2.0).
SCTYPES = {'int': [np.int8, np.int16, np.int32, np.int64],
           'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
           'float': [np.float16, np.float32, np.float64],
           'complex': [np.complex64, np.complex128],
           'others': [bool, object, bytes, str, np.void]}


def deprecate_with_doc(msg):
    # Adapted from: https://stackoverflow.com/a/30253848/1939576

    def dep(func):

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} deprecated, {msg}",
                category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return new_func

    return dep
