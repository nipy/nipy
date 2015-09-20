""" Testing compat3 module
"""
from __future__ import with_statement
from __future__ import absolute_import

from nibabel.py3k import asstr, asbytes

from ..compat3 import to_str, open4csv

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from nibabel.tmpdirs import InTemporaryDirectory


def test_to_str():
    # Test routine to convert to string
    assert_equal('1', to_str(1))
    assert_equal('1.0', to_str(1.0))
    assert_equal('from', to_str(asstr('from')))
    assert_equal('from', to_str(asbytes('from')))


def test_open4csv():
    # Test opening of csv files
    import csv
    contents = [['oh', 'my', 'G'],
                ['L', 'O', 'L'],
                ['when', 'cleaning', 'windas']]
    with InTemporaryDirectory():
        with open4csv('my.csv', 'w') as fobj:
            writer = csv.writer(fobj)
            writer.writerows(contents)
        with open4csv('my.csv', 'r') as fobj:
            dialect = csv.Sniffer().sniff(fobj.read())
            fobj.seek(0)
            reader = csv.reader(fobj, dialect)
            back = list(reader)
    assert_equal(contents, back)
    assert_raises(ValueError, open4csv, 'my.csv', 'rb')
    assert_raises(ValueError, open4csv, 'my.csv', 'wt')
