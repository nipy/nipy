"""
Compatibility module
"""

import warnings
warnings.warn(DeprecationWarning(
    "This module (nipy.neurospin.utils.mask) has been moved and " 
    "is depreciated. Please update your code to import from "
    "'nipy.neurospin.mask'."))

# Absolute import, as 'import *' doesnot work with relative imports
from nipy.neurospin.mask import *

