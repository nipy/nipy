''' These tests are for a bug in sympy or sympy use causing errors in
printing of numpy int64 values
'''

import numpy as np

import sympy

from nipy.modalities.fmri import formula, hrf, utils

from nose.tools import assert_true
from numpy.testing import assert_almost_equal
from nipy.testing import parametric


def test_formula_define():
    # Try and track down obscure sympy error with int64 printing
    onsets = np.array([20], dtype=np.int64)
    name = 'c0'
    # event formula, should result in something like:
    # evs = 0 + sympy.DiracDelta(-20 + utils.t)
    evs = utils.events(onsets)
    # at this point this:
    # c = formula.define(name, evs)
    # raises a very complicated error ending in:
    # File "/Users/mb312/usr/local/lib/python2.6/site-packages/sympy/core/numbers.py", line 519, in __new__
    # obj.q = int(q)
    # TypeError: __int__ returned non-int (type numpy.int64)
    an_int = sympy.Integer(19)
    # same line as above, still raises error:
    # c = formula.define(name, evs)
    # but, if we make '20' a sympy integer:
    an_int = sympy.Integer(20)
    # ok - now it works; caching error no?
    c = formula.define(name, evs)

