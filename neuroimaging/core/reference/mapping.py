"""
Mappings define a transformation between sets of coordinate systems.

These mappings can be used to transform between voxel space and real space,
for example.
"""

__docformat__ = 'restructuredtext'

import csv, copy
import urllib
from struct import unpack

import numpy as np
from numpy.linalg import inv, pinv

