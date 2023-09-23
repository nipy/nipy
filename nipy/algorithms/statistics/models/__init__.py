# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" models - Statistical Models
"""


__docformat__ = 'restructuredtext'

from nipy.testing import Tester

from . import glm, model, regression
from .info import __doc__

test = Tester().test
