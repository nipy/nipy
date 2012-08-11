""" Testing compat3 module
"""

from nibabel.py3k import asstr, asbytes

from ..compat3 import to_str

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_to_str():
    # Test routine to convert to string
    assert_equal('1', to_str(1))
    assert_equal('1.0', to_str(1.0))
    assert_equal('from', to_str(asstr('from')))
    assert_equal('from', to_str(asbytes('from')))
