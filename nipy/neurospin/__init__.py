# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""\
Neurospin functions and classes for nipy. 

(c) Copyright CEA-INRIA-INSERM, 2003-2009.
Distributed under the terms of the BSD License.

http://www.lnao.fr

functions for fMRI

This module contains several objects and functions for fMRI processing.
"""

from numpy.testing import Tester

# No subpackage should be imported here to avoid run-time errors
# related to missing dependencies or binary incompatibilities

test = Tester().test
bench = Tester().bench 
