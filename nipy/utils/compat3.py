""" Routines for Python 3 compatibility

These are in addition to the nibabel.py3k routines.
"""

import sys
py3 = sys.version_info[0] >= 3

if py3:
    def to_str(s):
        """ Convert `s` to string, decoding as latin1 if `s` is bytes
        """
        if isinstance(s, bytes):
            return s.decode('latin1')
        return str(s)
else:
    to_str = str


def open4csv(fname, mode):
    """ Open filename `fname` for CSV IO in read or write `mode`

    Parameters
    ----------
    fname : str
        filename to open
    mode : {'r', 'w'}
        Mode to open file.  Don't specify binary or text modes; we need to
        chose these according to python version.

    Returns
    -------
    fobj : file object
        open file object; needs to be closed by the caller
    """
    if mode not in ('r', 'w'):
        raise ValueError('Only "r" and "w" allowed for mode')
    if not py3: # Files for csv reading and writing should be binary mode
        return open(fname, mode + 'b')
    return open(fname, mode, newline='')
