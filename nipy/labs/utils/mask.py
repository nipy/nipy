"""
Compatibility module
"""

import warnings
warnings.warn(DeprecationWarning(
    "This module (nipy.labs.utils.mask) has been moved and "
    "is depreciated. Please update your code to import from "
    "'nipy.labs.mask'."))

# Absolute import, as 'import *' doesnot work with relative imports
from ..mask import *
