"""
Compatibility module
"""
from __future__ import absolute_import

import warnings
warnings.warn(DeprecationWarning(
    "This module (nipy.labs.utils.mask) has been moved and "
    "is depreciated. Please update your code to import from "
    "'nipy.labs.mask'."))

# Absolute import, as 'import *' doesnot work with relative imports
from nipy.labs.mask import *
