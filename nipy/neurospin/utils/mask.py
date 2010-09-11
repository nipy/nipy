"""
Compatibility module
"""

import warnings
warnings.warn(DeprecationWarning(
    "This module (nipy.neurospin.utils.mask) has been moved and " 
    "is depreciated. Please update your code to import from "
    "'nipy.neurospin.mask'."))

from ..mask import *

