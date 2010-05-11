"""\
Neurospin functions and classes for nipy. 

(c) Copyright CEA-INRIA-INSERM, 2003-2009.
Distributed under the terms of the BSD License.

http://www.lnao.fr

functions for fMRI

This module contains several objects and functions for fMRI processing.

"""

from numpy.testing import Tester

import statistical_mapping
from registration import register, transform 

test = Tester().test
bench = Tester().bench 
