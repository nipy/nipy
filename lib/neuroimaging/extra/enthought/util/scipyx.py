"""
scipyx imports either numpy or Numeric based scipy_base from old scipy.
"""
from numerix import which
_nx,_vx = which
if _nx=='numeric':
    try:
        import sys
        from scipy_base import *
        sys.modules['neuroimaging.extra.enthought.util.scipyx.fastumath'] = fastumath
        sys.modules['neuroimaging.extra.enthought.util.scipyx.limits'] = limits
        from scipy import polynomial
        
    except ImportError:
        print 'No scipy_base. Will assume numpy.'
        _nx='numpy'

elif _nx=='numpy':
    import sys
    try:
        from numpy.oldnumeric import *
    except ImportError:
        pass
    from numpy import *
    from numpy.core import umath as fastumath
    from scipy.misc import limits
    sys.modules['neuroimaging.extra.enthought.util.scipyx.fastumath'] = fastumath
    sys.modules['neuroimaging.extra.enthought.util.scipyx.limits'] = limits
    from numpy.lib import polynomial
else:
    print 'Nothing imported to scipyx'
