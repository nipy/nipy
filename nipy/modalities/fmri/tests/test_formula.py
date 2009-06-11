"""
Test functions for models.formula
"""

import string

import numpy as np
import numpy.random as R
import numpy.linalg as L
import numpy.testing as nptest
import nose.tools
import sympy

from nipy.modalities.fmri import formula as F
from nipy.modalities.fmri import aliased


def test_contrast1():

    x = F.Term('x')
    y = F.Term('y')
    z = F.Term('z')

    f = F.Formula([x,y])
    arr = F.make_recarray([[3,5,4],[8,21,-1],[4,6,-2]], 'xyz')

    D, C = f.design(arr, contrasts={'x':x.formula,
                                    'diff':F.Formula([x-y]),
                                    'sum':F.Formula([x+y]),
                                    'both':F.Formula([x-y,x+y])})
    yield nptest.assert_almost_equal, C['x'], np.array([1,0])
    yield nptest.assert_almost_equal, C['diff'], np.array([1,-1])
    yield nptest.assert_almost_equal, C['sum'], np.array([1,1])
    yield nptest.assert_almost_equal, C['both'], np.array([[1,-1],[1,1]])

    f = F.Formula([x,y,z])
    arr = F.make_recarray([[3,5,4],[8,21,-1],[4,6,-2]], 'xyz')

    D, C = f.design(arr, contrasts={'x':x.formula,
                                    'diff':F.Formula([x-y]),
                                    'sum':F.Formula([x+y]),
                                    'both':F.Formula([x-y,x+y])})
    yield nptest.assert_almost_equal, C['x'], np.array([1,0,0])
    yield nptest.assert_almost_equal, C['diff'], np.array([1,-1,0])
    yield nptest.assert_almost_equal, C['sum'], np.array([1,1,0])
    yield nptest.assert_almost_equal, C['both'], np.array([[1,-1,0],[1,1,0]])


def test_random_effects():
    subj = F.make_recarray([2,2,2,3,3], 's')
    subj_factor = F.Factor('s', [2,3])

    c = F.RandomEffects(subj_factor.terms, sigma=np.array([[4,1],[1,6]]))
    C = c.cov(subj)
    yield nptest.assert_almost_equal, C, [[4,4,4,1,1],
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
    yield nose.tools.assert_true, np.alltrue(t)


def test_design_expression():
    t1 = F.Term("x")
    t2 = F.Term('y')
    f = t1.formula + t2.formula
    nose.tools.assert_true(str(f.design_expr) in ['[x, y]', '[y, x]'])

###########################################################
"""
Tests for formula
"""
###########################################################


def test_formula_property():
    """
    Check that you can create a Formula with one term
    """
    t1 = F.Term("x")
    f = t1.formula
    nose.tools.assert_equal(f.design_expr, [t1])


def test_design():
    """
    Check that you get the design matrix we expect
    """
    t1 = F.Term("x")
    t2 = F.Term('y')

    n = F.make_recarray([2,4,5], 'x')
    yield nptest.assert_almost_equal, t1.formula.design(n)['x'], n['x']

    f = t1.formula + t2.formula
    n = F.make_recarray([(2,3),(4,5),(5,6)], 'xy')

    yield nptest.assert_almost_equal, f.design(n)['x'], n['x']
    yield nptest.assert_almost_equal, f.design(n)['y'], n['y']

    f = t1.formula + t2.formula + F.I + t1.formula * t2.formula
    yield nptest.assert_almost_equal, f.design(n)['x'], n['x']
    yield nptest.assert_almost_equal, f.design(n)['y'], n['y']
    yield nptest.assert_almost_equal, f.design(n)['1'], 1
    yield nptest.assert_almost_equal, f.design(n)['x*y'], n['x']*n['y']

    n = np.array([(2,3,'a'),(4,5,'b'),(5,6,'a')], np.dtype([('x', np.float),
                                                            ('y', np.float),
                                                            ('f', 'S1')]))
    f = F.Factor('f', ['a','b'])
    ff = t1.formula * f + F.I
    yield nptest.assert_almost_equal, ff.design(n)['f_a*x'], n['x']*[1,0,1]
    yield nptest.assert_almost_equal, ff.design(n)['f_b*x'], n['x']*[0,1,0]
    yield nptest.assert_almost_equal, ff.design(n)['1'], 1


def test_alias2():
    f = F.aliased_function('f', lambda x: 2*x)
    g = F.aliased_function('f', lambda x: np.sqrt(x))
    x = sympy.Symbol('x')

    l1 = aliased.lambdify(x, f(x))
    l2 = aliased.lambdify(x, g(x))

    yield nose.tools.assert_equal, str(f(x)), str(g(x))
    yield nose.tools.assert_equal, l1(3), 6
    yield nose.tools.assert_equal, l2(3), np.sqrt(3)


def test_alias():
    x = F.Term('x')
    f = F.aliased_function('f', lambda x: 2*x)
    g = F.aliased_function('g', lambda x: np.sqrt(x))

    ff = F.Formula([f(x), g(x)**2])
    n = F.make_recarray([2,4,5], 'x')
    yield nptest.assert_almost_equal, ff.design(n)['f(x)'], n['x']*2
    yield nptest.assert_almost_equal, ff.design(n)['g(x)**2'], n['x']


def test_nonlin1():
    # Fit an exponential curve, with the exponent stratified by a factor
    # with a common intercept and multiplicative factor in front of the
    # exponential
    x = F.Term('x')
    fac = F.Factor('f', 'ab')
    f = F.Formula([sympy.exp(fac.stratify(x).mean)]) + F.I

    params = F.getparams(f.mean)
    yield nose.tools.assert_equal, set([str(p) for p in params]), set(['_x0', '_x1', '_b0', '_b1'])
    
    test1 = set(['1',
                 'exp(_x0*f_a + _x1*f_b)',
                 '_b0*f_a*exp(_x0*f_a + _x1*f_b)',
                 '_b0*f_b*exp(_x0*f_a + _x1*f_b)'])
    test2 = set(['1',
                 'exp(_x0*f_a + _x1*f_b)',
                 '_b1*f_a*exp(_x0*f_a + _x1*f_b)',
                 '_b1*f_b*exp(_x0*f_a + _x1*f_b)'])
    yield nose.tools.assert_true, test1 or test2

    n = F.make_recarray([(2,3,'a'),(4,5,'b'),(5,6,'a')], 'xyf', ['d','d','S1'])
    p = F.make_recarray([1,2,3,4], ['_x0', '_x1', '_b0', '_b1'])
    A = f.design(n, p)
    print A, A.dtype


def test_Rintercept():

    x = F.Term('x')
    y = F.Term('x')
    xf = x.formula
    yf = y.formula
    newf = (xf+F.I)*(yf+F.I)
    nose.tools.assert_equal(set(newf.terms), set([x,y,x*y,sympy.Number(1)]))
    

def test_return_float():

    x = F.Term('x')
    f = F.Formula([x,x**2])
    xx= F.make_recarray(np.linspace(0,10,11), 'x')
    dtype = f.design(xx).dtype
    yield nose.tools.assert_equal, set(dtype.names), set(['x', 'x**2'])

    dtype = f.design(xx, return_float=True).dtype
    yield nose.tools.assert_equal, dtype, np.float


def test_natural_spline():

    xt=F.Term('x')

    ns=F.natural_spline(xt, knots=[2,6,9])
    xx= F.make_recarray(np.linspace(0,10,101), 'x')
    dd=ns.design(xx, return_float=True)
    xx = xx['x']
    yield nptest.assert_almost_equal, dd[:,0], xx
    yield nptest.assert_almost_equal, dd[:,1], xx**2
    yield nptest.assert_almost_equal, dd[:,2], xx**3
    yield nptest.assert_almost_equal, dd[:,3], (xx-2)**3*np.greater_equal(xx,2)
    yield nptest.assert_almost_equal, dd[:,4], (xx-6)**3*np.greater_equal(xx,6)
    yield nptest.assert_almost_equal, dd[:,5], (xx-9)**3*np.greater_equal(xx,9)

    ns=F.natural_spline(xt, knots=[2,9,6], intercept=True)
    xx= F.make_recarray(np.linspace(0,10,101), 'x')
    dd=ns.design(xx, return_float=True)
    xx = xx['x']
    yield nptest.assert_almost_equal, dd[:,0], 1
    yield nptest.assert_almost_equal, dd[:,1], xx
    yield nptest.assert_almost_equal, dd[:,2], xx**2
    yield nptest.assert_almost_equal, dd[:,3], xx**3
    yield nptest.assert_almost_equal, dd[:,4], (xx-2)**3*np.greater_equal(xx,2)
    yield nptest.assert_almost_equal, dd[:,5], (xx-9)**3*np.greater_equal(xx,9)
    yield nptest.assert_almost_equal, dd[:,6], (xx-6)**3*np.greater_equal(xx,6)




# class TestTerm:






#     def test_add(self):
#         t1 = F.Term("t1")
#         t2 = F.Term("t2")
#         f = t1 + t2
#         self.assert_(isinstance(f, F.Formula))
#         self.assert_(f.hasterm(t1))
#         self.assert_(f.hasterm(t2))

#     def test_mul(self):
#         t1 = F.Term("t1")
#         t2 = F.Term("t2")
#         f = t1 * t2
#         self.assert_(isinstance(f, F.Formula))

#         intercept = F.Term("intercept")
#         f = t1 * intercept
#         self.assertEqual(str(f), str(F.Formula(t1)))

#         f = intercept * t1
#         self.assertEqual(str(f), str(F.Formula(t1)))

# class TestFormula:

#     def setUp(self):
#         self.X = R.standard_normal((40,10))
#         self.namespace = {}
#         self.terms = []
#         for i in range(10):
#             name = '%s' % string.uppercase[i]
#             self.namespace[name] = self.X[:,i]
#             self.terms.append(F.Term(name))

#         self.formula = self.terms[0]
#         for i in range(1, 10):
#             self.formula += self.terms[i]
#         self.F.namespace = self.namespace

#     @dec.knownfailureif(True)
#     def test_namespace(self):
#         space1 = {'X':np.arange(50), 'Y':np.arange(50)*2}
#         space2 = {'X':np.arange(20), 'Y':np.arange(20)*2}
#         space3 = {'X':np.arange(30), 'Y':np.arange(30)*2}
#         X = F.Term('X')
#         Y = F.Term('Y')

#         X.namespace = space1
#         assert_almost_equal(X(), np.arange(50))

#         Y.namespace = space2
#         assert_almost_equal(Y(), np.arange(20)*2)

#         f = X + Y

#         f.namespace = space1
#         self.assertEqual(f().shape, (2,50))
#         assert_almost_equal(Y(), np.arange(20)*2)
#         assert_almost_equal(X(), np.arange(50))

#         f.namespace = space2
#         self.assertEqual(f().shape, (2,20))
#         assert_almost_equal(Y(), np.arange(20)*2)
#         assert_almost_equal(X(), np.arange(50))

#         f.namespace = space3
#         self.assertEqual(f().shape, (2,30))
#         assert_almost_equal(Y(), np.arange(20)*2)
#         assert_almost_equal(X(), np.arange(50))

#         xx = X**2
#         self.assertEqual(xx().shape, (50,))

#         xx.namespace = space3
#         self.assertEqual(xx().shape, (30,))

#         xx = X * F.I
#         self.assertEqual(xx().shape, (50,))
#         xx.namespace = space3
#         self.assertEqual(xx().shape, (30,))

#         xx = X * X
#         self.assertEqual(xx.namespace, X.namespace)

#         xx = X + Y
#         self.assertEqual(xx.namespace, {})

#         Y.namespace = {'X':np.arange(50), 'Y':np.arange(50)*2}
#         xx = X + Y
#         self.assertEqual(xx.namespace, {})

#         Y.namespace = X.namespace
#         xx = X+Y
#         self.assertEqual(xx.namespace, Y.namespace)

#     def test_termcolumns(self):
#         t1 = F.Term("A")
#         t2 = F.Term("B")
#         f = t1 + t2 + t1 * t2
#         def other(val):
#             return np.array([3.2*val,4.342*val**2, 5.234*val**3])
#         q = F.Quantitative(['other%d' % i for i in range(1,4)], termname='other', func=t1, transform=other)
#         f += q
#         q.namespace = f.namespace = self.F.namespace
#         assert_almost_equal(q(), f()[f.termcolumns(q)])


#     def test_str(self):
#         s = str(self.formula)

#     def test_call(self):
#         x = self.formula()
#         self.assertEquals(np.array(x).shape, (10, 40))

#     def test_design(self):
#         x = self.F.design()
#         self.assertEquals(x.shape, (40, 10))

#     def test_product(self):
#         prod = self.formula['A'] * self.formula['C']
#         f = self.formula + prod
#         f.namespace = self.namespace
#         x = f.design()
#         p = f['A*C']
#         p.namespace = self.namespace
#         col = f.termcolumns(prod, dict=False)
#         assert_almost_equal(np.squeeze(x[:,col]), self.X[:,0] * self.X[:,2])
#         assert_almost_equal(np.squeeze(p()), self.X[:,0] * self.X[:,2])

#     def test_intercept1(self):
#         prod = self.terms[0] * self.terms[2]
#         f = self.formula + F.I
#         icol = f.names().index('intercept')
#         f.namespace = self.namespace
#         assert_almost_equal(f()[icol], np.ones((40,)))

#     def test_intercept3(self):
#         t = self.formula['A']
#         t.namespace = self.namespace
#         prod = t * F.I
#         prod.namespace = self.F.namespace
#         assert_almost_equal(np.squeeze(prod()), t())

#     # FIXME: AttributeError: 'Contrast' object has no attribute 'getmatrix'
#     @dec.knownfailureif(True)
#     def test_contrast1(self):
#         term = self.terms[0] + self.terms[2]
#         c = contrast.Contrast(term, self.formula)
#         c.getmatrix()
#         col1 = self.F.termcolumns(self.terms[0], dict=False)
#         col2 = self.F.termcolumns(self.terms[1], dict=False)
#         test = [[1] + [0]*9, [0]*2 + [1] + [0]*7]
#         assert_almost_equal(c.matrix, test)

#     # FIXME: AttributeError: 'Contrast' object has no attribute 'getmatrix'
#     @dec.knownfailureif(True)
#     def test_contrast2(self):
#         dummy = F.Term('zero')
#         self.namespace['zero'] = np.zeros((40,), np.float64)
#         term = dummy + self.terms[2]
#         c = contrast.Contrast(term, self.formula)
#         c.getmatrix()
#         test = [0]*2 + [1] + [0]*7
#         assert_almost_equal(c.matrix, test)

#     # FIXME: AttributeError: 'Contrast' object has no attribute 'getmatrix'
#     @dec.knownfailureif(True)
#     def test_contrast3(self):
#         X = self.F.design()
#         P = np.dot(X, L.pinv(X))

#         dummy = F.Term('noise')
#         resid = np.identity(40) - P
#         self.namespace['noise'] = np.transpose(np.dot(resid, R.standard_normal((40,5))))
#         terms = dummy + self.terms[2]
#         terms.namespace = self.F.namespace
#         c = contrast.Contrast(terms, self.formula)
#         c.getmatrix()
#         self.assertEquals(c.matrix.shape, (10,))

#     def test_power(self):

#         t = self.terms[2]
#         t2 = t**2
#         t.namespace = t2.namespace = self.F.namespace
#         assert_almost_equal(t()**2, t2())

#     def test_quantitative(self):
#         t = self.terms[2]
#         sint = F.Quantitative('t', func=t, transform=np.sin)
#         t.namespace = sint.namespace = self.F.namespace
#         assert_almost_equal(np.sin(t()), sint())

#     def test_factor1(self):
#         f = ['a','b','c']*10
#         fac = F.Factor('ff', f)
#         fac.namespace = {'ff':f}
#         self.assertEquals(list(fac.values()), f)

#     def test_factor2(self):
#         f = ['a','b','c']*10
#         fac = F.Factor('ff', f)
#         fac.namespace = {'ff':f}
#         self.assertEquals(fac().shape, (3,30))

#     def test_factor3(self):
#         f = ['a','b','c']*10
#         fac = F.Factor('ff', f)
#         fac.namespace = {'ff':f}
#         m = fac.main_effect(reference=1)
#         m.namespace = fac.namespace
#         self.assertEquals(m().shape, (2,30))

#     def test_factor4(self):
#         f = ['a','b','c']*10
#         fac = F.Factor('ff', f)
#         fac.namespace = {'ff':f}
#         m = fac.main_effect(reference=2)
#         m.namespace = fac.namespace
#         r = np.array([np.identity(3)]*10)
#         r.shape = (30,3)
#         r = r.T
#         _m = np.array([r[0]-r[2],r[1]-r[2]])
#         assert_almost_equal(_m, m())

#     def test_factor5(self):
#         f = ['a','b','c']*3
#         fac = F.Factor('ff', f)
#         fac.namespace = {'ff':f}

#         assert_equal(fac(), [[1,0,0]*3,
#                              [0,1,0]*3,
#                              [0,0,1]*3])
#         assert_equal(fac['a'], [1,0,0]*3)
#         assert_equal(fac['b'], [0,1,0]*3)
#         assert_equal(fac['c'], [0,0,1]*3)


#     def test_ordinal_factor(self):
#         f = ['a','b','c']*3
#         fac = F.Factor('ff', ['a','b','c'], ordinal=True)
#         fac.namespace = {'ff':f}

#         assert_equal(fac(), [0,1,2]*3)
#         assert_equal(fac['a'], [1,0,0]*3)
#         assert_equal(fac['b'], [0,1,0]*3)
#         assert_equal(fac['c'], [0,0,1]*3)

#     def test_ordinal_factor2(self):
#         f = ['b','c', 'a']*3
#         fac = F.Factor('ff', ['a','b','c'], ordinal=True)
#         fac.namespace = {'ff':f}

#         assert_equal(fac(), [1,2,0]*3)
#         assert_equal(fac['a'], [0,0,1]*3)
#         assert_equal(fac['b'], [1,0,0]*3)
#         assert_equal(fac['c'], [0,1,0]*3)

#     # FIXME: AttributeError: 'Contrast' object has no attribute 'getmatrix'
#     @dec.knownfailureif(True)
#     def test_contrast4(self):

#         f = self.formula + self.terms[5] + self.terms[5]
#         f.namespace = self.namespace
#         estimable = False

#         c = contrast.Contrast(self.terms[5], f)
#         c.getmatrix()

#         self.assertEquals(estimable, False)

#     def test_interactions(self):

#         f = F.interactions([F.Term(l) for l in ['a', 'b', 'c']])
#         assert_equal(set(f.termnames()), set(['a', 'b', 'c', 'a*b', 'a*c', 'b*c']))

#         f = F.interactions([F.Term(l) for l in ['a', 'b', 'c', 'd']], order=3)
#         assert_equal(set(f.termnames()), set(['a', 'b', 'c', 'd', 'a*b', 'a*c', 'a*d', 'b*c', 'b*d', 'c*d', 'a*b*c', 'a*c*d', 'a*b*d', 'b*c*d']))

#         f = F.interactions([F.Term(l) for l in ['a', 'b', 'c', 'd']], order=[1,2,3])
#         assert_equal(set(f.termnames()), set(['a', 'b', 'c', 'd', 'a*b', 'a*c', 'a*d', 'b*c', 'b*d', 'c*d', 'a*b*c', 'a*c*d', 'a*b*d', 'b*c*d']))

#         f = F.interactions([F.Term(l) for l in ['a', 'b', 'c', 'd']], order=[3])
#         assert_equal(set(f.termnames()), set(['a*b*c', 'a*c*d', 'a*b*d', 'b*c*d']))

#     def test_subtract(self):
#         f = F.interactions([F.Term(l) for l in ['a', 'b', 'c']])
#         ff = f - f['a*b']
#         assert_equal(set(ff.termnames()), set(['a', 'b', 'c', 'a*c', 'b*c']))

#         ff = f - f['a*b'] - f['a*c']
#         assert_equal(set(ff.termnames()), set(['a', 'b', 'c', 'b*c']))

#         ff = f - (f['a*b'] + f['a*c'])
#         assert_equal(set(ff.termnames()), set(['a', 'b', 'c', 'b*c']))
