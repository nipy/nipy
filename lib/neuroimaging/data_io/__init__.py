"""
Package contains generic functions for data input/output. This includes
methods for accessing file systems and network resources.
"""

__docformat__ = 'restructuredtext'


__all__ = ["formats", "datasource"]

import formats
#, datasource

def test(level=1, verbosity=1, flags=[]):
    from neuroimaging.utils.testutils import set_flags
    set_flags(flags)
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
