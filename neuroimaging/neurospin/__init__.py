"""\
fff
==========

Functions for fMRI.

No decent documentation currently available.

(c) Copyright CEA-INRIA-INSERM, 2003-2008.

http://...
"""

__doc__ = """\
functions for fMRI

This module contains several objects and functions for fMRI processing.

No decent documentation currently available.

Distributed under the terms of the BSD License.

(c) Copyright CEA-INRIA-INSERM, 2003-2008. 
"""

import __config__
import bindings
#import utils 
#import glm
#import registration
#import neuro

try:
    from numpy.testing import Tester
except ImportError:
    from fff2.utils.nosetester import NoseTester as Tester
test = Tester().test
bench = Tester().bench 

