# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for the temporary matlab file module."""
from __future__ import absolute_import

# Stdlib imports
import os
import tempfile

# Our own imports
from nipy.interfaces.matlab import mlab_tempfile

# Functions, classes and other top-level code
def check_mlab_tempfile(dir):
    """Helper function for testing the mlab temp file creation."""

    try:
        f = mlab_tempfile(dir)
    except OSError as msg:
        if not os.path.isdir(dir) and 'No such file or directory' in msg:
            # This is OK, it's the expected error
            return True
        else:
            raise
    else:
        f.close()


def test_mlab_tempfile():
    for dir in [None,tempfile.tempdir,tempfile.mkdtemp()]:
        yield check_mlab_tempfile,dir
