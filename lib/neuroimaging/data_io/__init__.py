"""
Package contains generic functions for data input/output. This includes
methods for accessing file systems and network resources.
"""

__docformat__ = 'restructuredtext'

import formats
import datasource

__all__ = ["formats", "datasource"]

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
