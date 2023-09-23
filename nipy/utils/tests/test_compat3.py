""" Testing compat3 module
"""

from nibabel.tmpdirs import InTemporaryDirectory
from nose.tools import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from ..compat3 import open4csv, to_str


def test_to_str():
    # Test routine to convert to string
    assert '1' == to_str(1)
    assert '1.0' == to_str(1.0)
    assert 'from' == to_str('from')
    assert 'from' == to_str(b'from')


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
    assert contents == back
    assert_raises(ValueError, open4csv, 'my.csv', 'rb')
    assert_raises(ValueError, open4csv, 'my.csv', 'wt')
