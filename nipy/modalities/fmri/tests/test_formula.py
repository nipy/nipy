# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for models.formula
"""

import numpy as np
import sympy

from nipy.modalities.fmri import formula as F
from nipy.modalities.fmri import aliased

from nipy.testing import assert_almost_equal, assert_true, \
    assert_equal, assert_false, assert_raises, parametric


@parametric
def test_terms():
    t, = F.terms('a')
    a, b, c = F.Term('a'), F.Term('b'), F.Term('c')
    yield assert_equal(t, a)
    ts = F.terms('a', 'b', 'c')
    yield assert_equal(ts, (a, b, c))
    # a string without separator chars returns one symbol.  This is the
    # future sympy default. 
    yield assert_equal(F.terms('abc'), [F.Term('abc')])
    yield assert_equal(F.terms('a b c'), (a, b, c))
    yield assert_equal(F.terms('a, b, c'), (a, b, c))
    

def test_getparams_terms():
    t = F.Term('t')
    x, y, z = [sympy.Symbol(l) for l in 'xyz']
    yield assert_equal, set(F.getparams(x*y*t)), set([x,y])
    yield assert_equal, set(F.getterms(x*y*t)), set([t])

    matrix_expr = np.array([[x,y*t],[y,z]])
    yield assert_equal, set(F.getparams(matrix_expr)), set([x,y,z])
    yield assert_equal, set(F.getterms(matrix_expr)), set([t])
    

def test_formula_params():
    t = F.Term('t')
    x, y = [sympy.Symbol(l) for l in 'xy']
    f = F.Formula([t*x,y])
    yield assert_equal, set(f.params), set([x,y] + list(f.coefs.values()))


def test_contrast1():
    x = F.Term('x')
    yield assert_equal, x, x+x
    y = F.Term('y')
    z = F.Term('z')
    f = F.Formula([x,y])
    arr = F.make_recarray([[3,5,4],[8,21,-1],[4,6,-2]], 'xyz')
    D, C = f.design(arr, contrasts={'x':x.formula,
                                    'diff':F.Formula([x-y]),
                                    'sum':F.Formula([x+y]),
                                    'both':F.Formula([x-y,x+y])})
    yield assert_almost_equal, C['x'], np.array([1,0])
    yield assert_almost_equal, C['diff'], np.array([1,-1])
    yield assert_almost_equal, C['sum'], np.array([1,1])
    yield assert_almost_equal, C['both'], np.array([[1,-1],[1,1]])

    f = F.Formula([x,y,z])
    arr = F.make_recarray([[3,5,4],[8,21,-1],[4,6,-2]], 'xyz')
    D, C = f.design(arr, contrasts={'x':x.formula,
                                    'diff':F.Formula([x-y]),
                                    'sum':F.Formula([x+y]),
                                    'both':F.Formula([x-y,x+y])})
    yield assert_almost_equal, C['x'], np.array([1,0,0])
    yield assert_almost_equal, C['diff'], np.array([1,-1,0])
    yield assert_almost_equal, C['sum'], np.array([1,1,0])
    yield assert_almost_equal, C['both'], np.array([[1,-1,0],[1,1,0]])


def test_define():
    t = F.Term('t')
    expr = sympy.exp(3*t)
    yield assert_equal, str(expr), 'exp(3*t)'

    newf = F.define('f', expr)
    yield assert_equal, str(newf), 'f(t)'

    f = aliased.lambdify(t, newf)

    tval = np.random.standard_normal((3,))
    yield assert_almost_equal, np.exp(3*tval), f(tval)


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
    yield assert_equal, set([str(t) for t in f.terms]),  set(['x1', 'x2', 'x3', 'x4', 'x5', 'x6_green', 'x6_blue', 'x6_red'])
    yield assert_equal, set([str(t) for t in f.design_expr]),  set(['x1', 'x2', 'x3', 'x4', 'x5', 'x6_green', 'x6_blue', 'x6_red'])


def test_random_effects():
    subj = F.make_recarray([2,2,2,3,3], 's')
    subj_factor = F.Factor('s', [2,3])

    c = F.RandomEffects(subj_factor.terms, sigma=np.array([[4,1],[1,6]]))
    C = c.cov(subj)
    yield assert_almost_equal, C, [[4,4,4,1,1],
                                          [4,4,4,1,1],
                                          [4,4,4,1,1],
                                          [1,1,1,6,6],
                                          [1,1,1,6,6]]

    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = F.RandomEffects(subj_factor.terms, sigma=np.array([[a,0],[0,b]]))
    C = c.cov(subj)
    t = np.equal(C, [[a,a,a,0,0],
                     [a,a,a,0,0],
                     [a,a,a,0,0],
                     [0,0,0,b,b],
                     [0,0,0,b,b]])
    yield assert_true, np.alltrue(t)


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
    
    yield assert_equal, t2, t2*t2
    yield assert_equal, f, f*f
    yield assert_false, f == f2
    yield assert_equal, set((t2*x).atoms()), set([t2,x])


def test_make_recarray():
    m = F.make_recarray([[3,4],[4,6],[7,9]], 'wv', [np.float, np.int])

    yield assert_equal, m.dtype.names, ['w', 'v']

    m2 = F.make_recarray(m, 'xy')
    yield assert_equal, m2.dtype.names, ['x', 'y']

    
def test_str_formula():
    t1 = F.Term('x')
    t2 = F.Term('y')
    f = F.Formula([t1, t2])
    yield assert_equal, str(f), "Formula([x, y])"


def test_design():
    # Check that you get the design matrix we expect
    t1 = F.Term("x")
    t2 = F.Term('y')

    n = F.make_recarray([2,4,5], 'x')
    yield assert_almost_equal, t1.formula.design(n)['x'], n['x']

    f = t1.formula + t2.formula
    n = F.make_recarray([(2,3),(4,5),(5,6)], 'xy')

    yield assert_almost_equal, f.design(n)['x'], n['x']
    yield assert_almost_equal, f.design(n)['y'], n['y']

    f = t1.formula + t2.formula + F.I + t1.formula * t2.formula
    yield assert_almost_equal, f.design(n)['x'], n['x']
    yield assert_almost_equal, f.design(n)['y'], n['y']
    yield assert_almost_equal, f.design(n)['1'], 1
    yield assert_almost_equal, f.design(n)['x*y'], n['x']*n['y']
    # drop x field, check that design raises error
    ny = np.recarray(n.shape, dtype=[('x', n.dtype['x'])])
    ny['x'] = n['x']
    yield assert_raises, ValueError, f.design, ny
    n = np.array([(2,3,'a'),(4,5,'b'),(5,6,'a')], np.dtype([('x', np.float),
                                                            ('y', np.float),
                                                            ('f', 'S1')]))
    f = F.Factor('f', ['a','b'])
    ff = t1.formula * f + F.I
    yield assert_almost_equal, ff.design(n)['f_a*x'], n['x']*[1,0,1]
    yield assert_almost_equal, ff.design(n)['f_b*x'], n['x']*[0,1,0]
    yield assert_almost_equal, ff.design(n)['1'], 1


def test_alias2():
    f = F.aliased_function('f', lambda x: 2*x)
    g = F.aliased_function('f', lambda x: np.sqrt(x))
    x = sympy.Symbol('x')

    l1 = aliased.lambdify(x, f(x))
    l2 = aliased.lambdify(x, g(x))

    yield assert_equal, str(f(x)), str(g(x))
    yield assert_equal, l1(3), 6
    yield assert_equal, l2(3), np.sqrt(3)


def test_alias():
    x = F.Term('x')
    f = F.aliased_function('f', lambda x: 2*x)
    g = F.aliased_function('g', lambda x: np.sqrt(x))

    ff = F.Formula([f(x), g(x)**2])
    n = F.make_recarray([2,4,5], 'x')
    yield assert_almost_equal, ff.design(n)['f(x)'], n['x']*2
    yield assert_almost_equal, ff.design(n)['g(x)**2'], n['x']


def test_factor_getterm():
    fac = F.Factor('f', 'ab')
    yield assert_equal, fac['f_a'], fac.get_term('a')
    fac = F.Factor('f', [1,2])
    yield assert_equal, fac['f_1'], fac.get_term(1)
    fac = F.Factor('f', [1,2])
    yield assert_raises, ValueError, fac.get_term, '1'
    m = fac.main_effect
    yield assert_equal, set(m.terms), set([fac['f_1']-fac['f_2']])

    
def test_stratify():
    fac = F.Factor('x', [2,3])

    y = sympy.Symbol('y')
    f = sympy.Function('f')
    yield assert_raises, ValueError, fac.stratify, f(y)


def test_fullrank():
    X = np.random.standard_normal((40,5))
    X[:,0] = X[:,1] + X[:,2]

    Y1 = F.fullrank(X)
    yield assert_equal, Y1.shape, (40,4)

    Y2 = F.fullrank(X, r=3)
    yield assert_equal, Y2.shape, (40,3)

    Y3 = F.fullrank(X, r=4)
    yield assert_equal, Y3.shape, (40,4)

    yield assert_almost_equal, Y1, Y3


def test_nonlin1():
    # Fit an exponential curve, with the exponent stratified by a factor
    # with a common intercept and multiplicative factor in front of the
    # exponential
    x = F.Term('x')
    fac = F.Factor('f', 'ab')
    f = F.Formula([sympy.exp(fac.stratify(x).mean)]) + F.I
    params = F.getparams(f.mean)
    yield assert_equal, set([str(p) for p in params]), set(['_x0', '_x1', '_b0', '_b1'])
    test1 = set(['1',
                 'exp(_x0*f_a + _x1*f_b)',
                 '_b0*f_a*exp(_x0*f_a + _x1*f_b)',
                 '_b0*f_b*exp(_x0*f_a + _x1*f_b)'])
    test2 = set(['1',
                 'exp(_x0*f_a + _x1*f_b)',
                 '_b1*f_a*exp(_x0*f_a + _x1*f_b)',
                 '_b1*f_b*exp(_x0*f_a + _x1*f_b)'])
    yield assert_true, test1 or test2
    n = F.make_recarray([(2,3,'a'),(4,5,'b'),(5,6,'a')], 'xyf', ['d','d','S1'])
    p = F.make_recarray([1,2,3,4], ['_x0', '_x1', '_b0', '_b1'])
    A = f.design(n, p)
    print A, A.dtype


def test_intercept():
    dz = F.make_recarray([2,3,4],'z')
    v = F.I.design(dz, return_float=False)
    yield assert_equal, v.dtype.names, ['intercept']


def test_nonlin2():
    dz = F.make_recarray([2,3,4],'z')
    z = F.Term('z')
    t = sympy.Symbol('th')
    p = F.make_recarray([3], ['tt'])
    f = F.Formula([sympy.exp(t*z)])
    yield assert_raises, ValueError, f.design, dz, p


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
    yield assert_equal, set(dtype.names), set(['x', 'x**2'])
    dtype = f.design(xx, return_float=True).dtype
    yield assert_equal, dtype, np.float


def test_subtract():
    x, y, z = [F.Term(l) for l in 'xyz']

    f1 = F.Formula([x,y])
    f2 = F.Formula([x,y,z])

    f3 = f2 - f1

    yield assert_equal, set(f3.terms), set([z])
    
    f4 = F.Formula([y,z])
    f5 = f1 - f4
    yield assert_equal, set(f5.terms), set([x])


def test_subs():
    t1 = F.Term("x")
    t2 = F.Term('y')
    z = F.Term('z')
    f = F.Formula([t1, t2])
    g = f.subs(t1, z)

    yield assert_equal, list(g.terms), [z, t2]


def test_natural_spline():
    xt=F.Term('x')

    ns=F.natural_spline(xt, knots=[2,6,9])
    xx= F.make_recarray(np.linspace(0,10,101), 'x')
    dd=ns.design(xx, return_float=True)
    xx = xx['x']
    yield assert_almost_equal, dd[:,0], xx
    yield assert_almost_equal, dd[:,1], xx**2
    yield assert_almost_equal, dd[:,2], xx**3
    yield assert_almost_equal, dd[:,3], (xx-2)**3*np.greater_equal(xx,2)
    yield assert_almost_equal, dd[:,4], (xx-6)**3*np.greater_equal(xx,6)
    yield assert_almost_equal, dd[:,5], (xx-9)**3*np.greater_equal(xx,9)

    ns=F.natural_spline(xt, knots=[2,9,6], intercept=True)
    xx= F.make_recarray(np.linspace(0,10,101), 'x')
    dd=ns.design(xx, return_float=True)
    xx = xx['x']
    yield assert_almost_equal, dd[:,0], 1
    yield assert_almost_equal, dd[:,1], xx
    yield assert_almost_equal, dd[:,2], xx**2
    yield assert_almost_equal, dd[:,3], xx**3
    yield assert_almost_equal, dd[:,4], (xx-2)**3*np.greater_equal(xx,2)
    yield assert_almost_equal, dd[:,5], (xx-9)**3*np.greater_equal(xx,9)
    yield assert_almost_equal, dd[:,6], (xx-6)**3*np.greater_equal(xx,6)
