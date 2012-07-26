""" Routines for Python 3 compatibility

These are in addition to the nibabel.py3k routines.
"""

import sys

if sys.version_info[0] >= 3:
    def to_str(s):
        """ Convert `s` to string, decoding as latin1 if `s` is bytes
        """
        if isinstance(s, bytes):
            return s.decode('latin1')
        return str(s)
else:
    to_str = str
