# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for formulae
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import sympy

from sympy.utilities.lambdify import implemented_function

from .. import formulae as F
from ..formulae import terms, Term

from nibabel.py3k import asbytes

from nose.tools import (assert_true, assert_equal, assert_false,
                        assert_raises)

from numpy.testing import assert_almost_equal, assert_array_equal

def test_terms():
    t = terms('a')
    assert_true(isinstance(t, Term))
    a, b, c = Term('a'), Term('b'), Term('c')
    assert_equal(t, a)
    ts = terms(('a', 'b', 'c'))
    assert_equal(ts, (a, b, c))
    # a string without separator chars returns one symbol.  This is the
    # sympy 0.7 behavior
    assert_equal(terms('abc'), Term('abc'))
    # separators return multiple symbols
    assert_equal(terms('a b c'), (a, b, c))
    assert_equal(terms('a, b, c'), (a, b, c))
    # no arg is an error
    assert_raises(TypeError, terms)
    # but empty arg returns empty tuple
    assert_equal(terms(()), ())
    # Test behavior of deprecated each_char kwarg
    assert_raises(TypeError, terms, 'abc', each_char=True)


def test_getparams_terms():
    t = F.Term('t')
    x, y, z = [sympy.Symbol(l) for l in 'xyz']
    assert_equal(set(F.getparams(x*y*t)), set([x,y]))
    assert_equal(set(F.getterms(x*y*t)), set([t]))

    matrix_expr = np.array([[x,y*t],[y,z]])
    assert_equal(set(F.getparams(matrix_expr)), set([x,y,z]))
    assert_equal(set(F.getterms(matrix_expr)), set([t]))


def test_formula_params():
    t = F.Term('t')
    x, y = [sympy.Symbol(l) for l in 'xy']
    f = F.Formula([t*x,y])
    assert_equal(set(f.params), set([x,y] + list(f.coefs.values())))


def test_contrast1():
    x = F.Term('x')
    assert_equal(x, x+x)
    y = F.Term('y')
    z = F.Term('z')
    f = F.Formula([x,y])
    arr = F.make_recarray([[3,5,4],[8,21,-1],[4,6,-2]], 'xyz')
    D, C = f.design(arr, contrasts={'x':x.formula,
                                    'diff':F.Formula([x-y]),
                                    'sum':F.Formula([x+y]),
                                    'both':F.Formula([x-y,x+y])})
    assert_almost_equal(C['x'], np.array([1,0]))
    assert_almost_equal(C['diff'], np.array([1,-1]))
    assert_almost_equal(C['sum'], np.array([1,1]))
    assert_almost_equal(C['both'], np.array([[1,-1],[1,1]]))

    f = F.Formula([x,y,z])
    arr = F.make_recarray([[3,5,4],[8,21,-1],[4,6,-2]], 'xyz')
    D, C = f.design(arr, contrasts={'x':x.formula,
                                    'diff':F.Formula([x-y]),
                                    'sum':F.Formula([x+y]),
                                    'both':F.Formula([x-y,x+y])})
    assert_almost_equal(C['x'], np.array([1,0,0]))
    assert_almost_equal(C['diff'], np.array([1,-1,0]))
    assert_almost_equal(C['sum'], np.array([1,1,0]))
    assert_almost_equal(C['both'], np.array([[1,-1,0],[1,1,0]]))


def test_formula_from_recarray():
    D = np.rec.array([
            (43, 51, 30, 39, 61, 92, 'blue'),
            (63, 64, 51, 54, 63, 73, 'blue'),
            (71, 70, 68, 69, 76, 86, 'red'),
            (61, 63, 45, 47, 54, 84, 'red'),
            (81, 78, 56, 66, 71, 83, 'blue'),
            (43, 55, 49, 44, 54, 49, 'blue'),
            (58, 67, 42, 56, 66, 68, 'green'),
            (71, 75, 50, 55, 70, 66, 'green'),
            (72, 82, 72, 67, 71, 83, 'blue'),
            (67, 61, 45, 47, 62, 80, 'red'),
            (64, 53, 53, 58, 58, 67, 'blue'),
            (67, 60, 47, 39, 59, 74, 'green'),
            (69, 62, 57, 42, 55, 63, 'blue'),
            (68, 83, 83, 45, 59, 77, 'red'),
            (77, 77, 54, 72, 79, 77, 'red'),
            (81, 90, 50, 72, 60, 54, 'blue'),
            (74, 85, 64, 69, 79, 79, 'green'),
            (65, 60, 65, 75, 55, 80, 'green'),
            (65, 70, 46, 57, 75, 85, 'red'),
            (50, 58, 68, 54, 64, 78, 'red'),
            (50, 40, 33, 34, 43, 64, 'blue'),
            (64, 61, 52, 62, 66, 80, 'blue'),
            (53, 66, 52, 50, 63, 80, 'red'),
            (40, 37, 42, 58, 50, 57, 'red'),
            (63, 54, 42, 48, 66, 75, 'blue'),
            (66, 77, 66, 63, 88, 76, 'blue'),
            (78, 75, 58, 74, 80, 78, 'red'),
            (48, 57, 44, 45, 51, 83, 'blue'),
            (85, 85, 71, 71, 77, 74, 'red'),
            (82, 82, 39, 59, 64, 78, 'blue')], 
                     dtype=[('y', 'i8'),
                            ('x1', 'i8'),
                            ('x2', 'i8'),
                            ('x3', 'i8'),
                            ('x4', 'i8'),
                            ('x5', 'i8'),
                            ('x6', '|S5')])
    f = F.Formula.fromrec(D, drop='y')
    assert_equal(set([str(t) for t in f.terms]),
                 set(['x1', 'x2', 'x3', 'x4', 'x5',
                      'x6_green', 'x6_blue', 'x6_red']))
    assert_equal(set([str(t) for t in f.design_expr]),
                 set(['x1', 'x2', 'x3', 'x4', 'x5',
                      'x6_green', 'x6_blue', 'x6_red']))


def test_random_effects():
    subj = F.make_recarray([2,2,2,3,3], 's')
    subj_factor = F.Factor('s', [2,3])

    c = F.RandomEffects(subj_factor.terms, sigma=np.array([[4,1],[1,6]]))
    C = c.cov(subj)
    assert_almost_equal(C, [[4,4,4,1,1],
                            [4,4,4,1,1],
                            [4,4,4,1,1],
                            [1,1,1,6,6],
                            [1,1,1,6,6]])
    # Sympy 0.7.0 does not cancel 1.0 * A to A; however, the dot product in the
    # covariance calculation returns floats, which are them multiplied by the
    # terms to give term * 1.0, etc.  We just insert the annoying floating point
    # here for the test, relying on sympy to do the same thing here as in the
    # dot product
    a = sympy.Symbol('a') * 1.0
    b = sympy.Symbol('b') * 1.0
    c = F.RandomEffects(subj_factor.terms, sigma=np.array([[a,0],[0,b]]))
    C = c.cov(subj)
    t = np.equal(C, [[a,a,a,0,0],
                     [a,a,a,0,0],
                     [a,a,a,0,0],
                     [0,0,0,b,b],
                     [0,0,0,b,b]])
    assert_true(np.alltrue(t))


def test_design_expression():
    t1 = F.Term("x")
    t2 = F.Term('y')
    f = t1.formula + t2.formula
    assert_true(str(f.design_expr) in ['[x, y]', '[y, x]'])


def test_formula_property():
    # Check that you can create a Formula with one term
    t1 = F.Term("x")
    f = t1.formula
    assert_equal(f.design_expr, [t1])


def test_mul():
    f = F.Factor('t', [2,3])
    f2 = F.Factor('t', [2,3,4])
    t2 = f['t_2']
    x = F.Term('x')
    assert_equal(t2, t2*t2)
    assert_equal(f, f*f)
    assert_false(f == f2)
    assert_equal(set((t2*x).atoms()), set([t2,x]))


def test_factor_add_sub():
    # Test adding and subtracting Factors
    f1 = F.Factor('t', [2, 3, 4])
    f2 = F.Factor('t', [2, 3])
    # Terms do not cancel in addition
    assert_equal(f1 + f2, F.Formula(np.hstack((f1.terms, f2.terms))))
    assert_equal(f1 - f2, F.Factor('t', [4]))
    f3 = F.Factor('p', [0, 1])
    assert_equal(f1 + f3, F.Formula(np.hstack((f1.terms, f3.terms))))
    assert_equal(f1 - f3, f1)


def test_term_order_sub():
    # Test preservation of term order in subtraction
    f1 = F.Formula(terms('z, y, x, w'))
    f2 = F.Formula(terms('x, y, a'))
    assert_array_equal((f1 - f2).terms, terms('z, w'))
    assert_array_equal((f2 - f1).terms, terms('a'))


def test_make_recarray():
    m = F.make_recarray([[3,4],[4,6],[7,9]], 'wv', [np.float, np.int])
    assert_equal(m.dtype.names, ('w', 'v'))
    m2 = F.make_recarray(m, 'xy')
    assert_equal(m2.dtype.names, ('x', 'y'))


def test_str_formula():
    t1 = F.Term('x')
    t2 = F.Term('y')
    f = F.Formula([t1, t2])
    assert_equal(str(f), "Formula([x, y])")


def test_design():
    # Check that you get the design matrix we expect
    t1 = F.Term("x")
    t2 = F.Term('y')

    n = F.make_recarray([2,4,5], 'x')
    assert_almost_equal(t1.formula.design(n)['x'], n['x'])

    f = t1.formula + t2.formula
    n = F.make_recarray([(2,3),(4,5),(5,6)], 'xy')

    assert_almost_equal(f.design(n)['x'], n['x'])
    assert_almost_equal(f.design(n)['y'], n['y'])

    f = t1.formula + t2.formula + F.I + t1.formula * t2.formula
    assert_almost_equal(f.design(n)['x'], n['x'])
    assert_almost_equal(f.design(n)['y'], n['y'])
    assert_almost_equal(f.design(n)['1'], 1)
    assert_almost_equal(f.design(n)['x*y'], n['x']*n['y'])
    # drop x field, check that design raises error
    ny = np.recarray(n.shape, dtype=[('x', n.dtype['x'])])
    ny['x'] = n['x']
    assert_raises(ValueError, f.design, ny)
    n = np.array([(2,3,'a'),(4,5,'b'),(5,6,'a')], np.dtype([('x', np.float),
                                                            ('y', np.float),
                                                            ('f', 'S1')]))
    f = F.Factor('f', ['a','b'])
    ff = t1.formula * f + F.I
    assert_almost_equal(ff.design(n)['f_a*x'], n['x']*[1,0,1])
    assert_almost_equal(ff.design(n)['f_b*x'], n['x']*[0,1,0])
    assert_almost_equal(ff.design(n)['1'], 1)


def test_design_inputs():
    # Check we can send in fields of type 'S', 'U', 'O' for design
    regf = F.Formula(F.terms('x, y'))
    f = F.Factor('f', ['a', 'b'])
    ff = regf + f
    for field_type in ('S1', 'U1', 'O'):
        data = np.array([(2, 3, 'a'),
                         (4, 5, 'b'),
                         (5, 6, 'a')],
                        dtype = [('x', np.float),
                                 ('y', np.float),
                                 ('f', field_type)])
        assert_array_equal(ff.design(data, return_float=True),
                           [[2, 3, 1, 0],
                            [4, 5, 0, 1],
                            [5, 6, 1, 0]])


def test_formula_inputs():
    # Check we can send in fields of type 'S', 'U', 'O' for factor levels
    level_names = ['red', 'green', 'blue']
    for field_type in ('S', 'U', 'O'):
        levels = np.array(level_names, dtype=field_type)
        f = F.Factor('myname', levels)
        assert_equal(f.levels, level_names)
    # Sending in byte objects
    levels = [asbytes(L) for L in level_names]
    f = F.Factor('myname', levels)
    assert_equal(f.levels, level_names)


def test_alias():
    x = F.Term('x')
    f = implemented_function('f', lambda x: 2*x)
    g = implemented_function('g', lambda x: np.sqrt(x))
    ff = F.Formula([f(x), g(x)**2])
    n = F.make_recarray([2,4,5], 'x')
    assert_almost_equal(ff.design(n)['f(x)'], n['x']*2)
    assert_almost_equal(ff.design(n)['g(x)**2'], n['x'])


def test_factor_getterm():
    fac = F.Factor('f', 'ab')
    assert_equal(fac['f_a'], fac.get_term('a'))
    fac = F.Factor('f', [1,2])
    assert_equal(fac['f_1'], fac.get_term(1))
    fac = F.Factor('f', [1,2])
    assert_raises(ValueError, fac.get_term, '1')
    m = fac.main_effect
    assert_equal(set(m.terms), set([fac['f_1']-fac['f_2']]))


def test_stratify():
    fac = F.Factor('x', [2,3])

    y = sympy.Symbol('y')
    f = sympy.Function('f')
    assert_raises(ValueError, fac.stratify, f(y))


def test_nonlin1():
    # Fit an exponential curve, with the exponent stratified by a factor
    # with a common intercept and multiplicative factor in front of the
    # exponential
    x = F.Term('x')
    fac = F.Factor('f', 'ab')
    f = F.Formula([sympy.exp(fac.stratify(x).mean)]) + F.I
    params = F.getparams(f.mean)
    assert_equal(set([str(p) for p in params]),
                 set(['_x0', '_x1', '_b0', '_b1']))
    test1 = set(['1',
                 'exp(_x0*f_a + _x1*f_b)',
                 '_b0*f_a*exp(_x0*f_a + _x1*f_b)',
                 '_b0*f_b*exp(_x0*f_a + _x1*f_b)'])
    test2 = set(['1',
                 'exp(_x0*f_a + _x1*f_b)',
                 '_b1*f_a*exp(_x0*f_a + _x1*f_b)',
                 '_b1*f_b*exp(_x0*f_a + _x1*f_b)'])
    assert_true(test1 or test2)
    n = F.make_recarray([(2,3,'a'),(4,5,'b'),(5,6,'a')], 'xyf', ['d','d','S1'])
    p = F.make_recarray([1,2,3,4], ['_x0', '_x1', '_b0', '_b1'])
    A = f.design(n, p)
    print(A, A.dtype)


def test_intercept():
    dz = F.make_recarray([2,3,4],'z')
    v = F.I.design(dz, return_float=False)
    assert_equal(v.dtype.names, ('intercept',))


def test_nonlin2():
    dz = F.make_recarray([2,3,4],'z')
    z = F.Term('z')
    t = sympy.Symbol('th')
    p = F.make_recarray([3], ['tt'])
    f = F.Formula([sympy.exp(t*z)])
    assert_raises(ValueError, f.design, dz, p)


def test_Rintercept():
    x = F.Term('x')
    y = F.Term('x')
    xf = x.formula
    yf = y.formula
    newf = (xf+F.I)*(yf+F.I)
    assert_equal(set(newf.terms), set([x,y,x*y,sympy.Number(1)]))


def test_return_float():
    x = F.Term('x')
    f = F.Formula([x,x**2])
    xx= F.make_recarray(np.linspace(0,10,11), 'x')
    dtype = f.design(xx).dtype
    assert_equal(set(dtype.names), set(['x', 'x**2']))
    dtype = f.design(xx, return_float=True).dtype
    assert_equal(dtype, np.float)


def test_subtract():
    x, y, z = [F.Term(l) for l in 'xyz']
    f1 = F.Formula([x,y])
    f2 = F.Formula([x,y,z])
    f3 = f2 - f1
    assert_equal(set(f3.terms), set([z]))
    f4 = F.Formula([y,z])
    f5 = f1 - f4
    assert_equal(set(f5.terms), set([x]))


def test_subs():
    t1 = F.Term("x")
    t2 = F.Term('y')
    z = F.Term('z')
    f = F.Formula([t1, t2])
    g = f.subs(t1, z)
    assert_equal(list(g.terms), [z, t2])


def test_natural_spline():
    xt=F.Term('x')

    ns=F.natural_spline(xt, knots=[2,6,9])
    xx= F.make_recarray(np.linspace(0,10,101), 'x')
    dd=ns.design(xx, return_float=True)
    xx = xx['x']
    assert_almost_equal(dd[:,0], xx)
    assert_almost_equal(dd[:,1], xx**2)
    assert_almost_equal(dd[:,2], xx**3)
    assert_almost_equal(dd[:,3], (xx-2)**3*np.greater_equal(xx,2))
    assert_almost_equal(dd[:,4], (xx-6)**3*np.greater_equal(xx,6))
    assert_almost_equal(dd[:,5], (xx-9)**3*np.greater_equal(xx,9))

    ns=F.natural_spline(xt, knots=[2,9,6], intercept=True)
    xx= F.make_recarray(np.linspace(0,10,101), 'x')
    dd=ns.design(xx, return_float=True)
    xx = xx['x']
    assert_almost_equal(dd[:,0], 1)
    assert_almost_equal(dd[:,1], xx)
    assert_almost_equal(dd[:,2], xx**2)
    assert_almost_equal(dd[:,3], xx**3)
    assert_almost_equal(dd[:,4], (xx-2)**3*np.greater_equal(xx,2))
    assert_almost_equal(dd[:,5], (xx-9)**3*np.greater_equal(xx,9))
    assert_almost_equal(dd[:,6], (xx-6)**3*np.greater_equal(xx,6))


def test_factor_term():
    # Test that byte strings, unicode strings and objects convert correctly
    for nt in 'S3', 'U3', 'O':
        ndt = np.dtype(nt)
        for lt in 'S3', 'U3', 'O':
            ldt = np.dtype(lt)
            name = np.asscalar(np.array('foo', ndt))
            level = np.asscalar(np.array('bar', ldt))
            ft = F.FactorTerm(name, level)
            assert_equal(str(ft), 'foo_bar')
