# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" models - Statistical Models
"""

from __future__ import absolute_import

__docformat__ = 'restructuredtext'

from .info import __doc__

from . import model
from . import regression
from . import glm

from nipy.testing import Tester
test = Tester().test
