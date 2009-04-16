"""\
Neurospin functions and classes for nipy. 

No decent documentation currently available.

(c) Copyright CEA-INRIA-INSERM, 2003-2009.

http://www.lnao.fr
"""

__doc__ = """\
functions for fMRI

This module contains several objects and functions for fMRI processing.

No decent documentation currently available.

Distributed under the terms of the BSD License.

(c) Copyright CEA-INRIA-INSERM, 2003-2008. 
"""

from numpy.testing import Tester

from image import Image 
import image_registration
import statistical_mapping

"""
import bindings
import glm
import register
import utils 
"""

test = Tester().test
bench = Tester().bench 
