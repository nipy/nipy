# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The Image class provides the interface which should be used
by users at the application level. The image provides a coordinate map,
and the data itself.

"""
from __future__ import absolute_import
__docformat__ = 'restructuredtext'


from nipy.testing import Tester
test = Tester().test
bench = Tester().bench

