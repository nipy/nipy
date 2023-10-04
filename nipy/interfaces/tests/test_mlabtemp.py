# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for the temporary matlab file module."""

# Stdlib imports
import os
import tempfile

import pytest

# Our own imports
from nipy.interfaces.matlab import mlab_tempfile


def test_mlab_tempfile():
    for dir in [None, tempfile.tempdir, tempfile.mkdtemp()]:
        assert mlab_tempfile(dir)
    pytest.raises(OSError, mlab_tempfile, 'non-existent-dir')
