# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

'''
Formula objects
===============

A formula is basically a sympy expression for the mean of something of
the form::

   mean = sum([Beta(e)*e for e in expr])

Or, a linear combination of sympy expressions, with each one multiplied
by its own "Beta". The elements of expr can be instances of Term (for a
linear regression formula, they would all be instances of Term). But, in
general, there might be some other parameters (i.e. sympy.Symbol
instances) that are not Terms.

The design matrix is made up of columns that are the derivatives of mean
with respect to everything that is not a Term, evaluted at a recarray
that has field names given by [str(t) for t in self.terms].

For those familiar with R's formula syntax, if we wanted a design matrix
like the following::

    > s.table = read.table("http://www-stat.stanford.edu/~jtaylo/courses/stats191/data/supervisor.table", header=T)
    > d = model.matrix(lm(Y ~ X1*X3, s.table)
    )
    > d
       (Intercept) X1 X3 X1:X3
    1            1 51 39  1989
    2            1 64 54  3456
    3            1 70 69  4830
    4            1 63 47  2961
    5            1 78 66  5148
    6            1 55 44  2420
    7            1 67 56  3752
    8            1 75 55  4125
    9            1 82 67  5494
    10           1 61 47  2867
    11           1 53 58  3074
    12           1 60 39  2340
    13           1 62 42  2604
    14           1 83 45  3735
    15           1 77 72  5544
    16           1 90 72  6480
    17           1 85 69  5865
    18           1 60 75  4500
    19           1 70 57  3990
    20           1 58 54  3132
    21           1 40 34  1360
    22           1 61 62  3782
    23           1 66 50  3300
    24           1 37 58  2146
    25           1 54 48  2592
    26           1 77 63  4851
    27           1 75 74  5550
    28           1 57 45  2565
    29           1 85 71  6035
    30           1 82 59  4838
    attr(,"assign")
    [1] 0 1 2 3
    >

With the Formula, it looks like this:

>>> r = np.rec.array([
...     (43, 51, 30, 39, 61, 92, 45), (63, 64, 51, 54, 63, 73, 47), 
...     (71, 70, 68, 69, 76, 86, 48), (61, 63, 45, 47, 54, 84, 35),
...     (81, 78, 56, 66, 71, 83, 47), (43, 55, 49, 44, 54, 49, 34),
...     (58, 67, 42, 56, 66, 68, 35), (71, 75, 50, 55, 70, 66, 41),
...     (72, 82, 72, 67, 71, 83, 31), (67, 61, 45, 47, 62, 80, 41),
...     (64, 53, 53, 58, 58, 67, 34), (67, 60, 47, 39, 59, 74, 41),
...     (69, 62, 57, 42, 55, 63, 25), (68, 83, 83, 45, 59, 77, 35),
...     (77, 77, 54, 72, 79, 77, 46), (81, 90, 50, 72, 60, 54, 36),
...     (74, 85, 64, 69, 79, 79, 63), (65, 60, 65, 75, 55, 80, 60),
...     (65, 70, 46, 57, 75, 85, 46), (50, 58, 68, 54, 64, 78, 52),
...     (50, 40, 33, 34, 43, 64, 33), (64, 61, 52, 62, 66, 80, 41),
...     (53, 66, 52, 50, 63, 80, 37), (40, 37, 42, 58, 50, 57, 49),
...     (63, 54, 42, 48, 66, 75, 33), (66, 77, 66, 63, 88, 76, 72),
...     (78, 75, 58, 74, 80, 78, 49), (48, 57, 44, 45, 51, 83, 38),
...     (85, 85, 71, 71, 77, 74, 55), (82, 82, 39, 59, 64, 78, 39)],
...              dtype=[('y', '<i8'), ('x1', '<i8'), ('x2', '<i8'),
...                     ('x3', '<i8'), ('x4', '<i8'), ('x5', '<i8'),
...                     ('x6', '<i8')])
>>> x1 = Term('x1'); x3 = Term('x3')
>>> f = Formula([x1, x3, x1*x3]) + I
>>> f.mean
_b0*x1 + _b1*x3 + _b2*x1*x3 + _b3

The I is the "intercept" term, I have explicity not used R's default of
adding it to everything.

>>> f.design(r)
array([(51.0, 39.0, 1989.0, 1.0), (64.0, 54.0, 3456.0, 1.0),
       (70.0, 69.0, 4830.0, 1.0), (63.0, 47.0, 2961.0, 1.0),
       (78.0, 66.0, 5148.0, 1.0), (55.0, 44.0, 2420.0, 1.0),
       (67.0, 56.0, 3752.0, 1.0), (75.0, 55.0, 4125.0, 1.0),
       (82.0, 67.0, 5494.0, 1.0), (61.0, 47.0, 2867.0, 1.0),
       (53.0, 58.0, 3074.0, 1.0), (60.0, 39.0, 2340.0, 1.0),
       (62.0, 42.0, 2604.0, 1.0), (83.0, 45.0, 3735.0, 1.0),
       (77.0, 72.0, 5544.0, 1.0), (90.0, 72.0, 6480.0, 1.0),
       (85.0, 69.0, 5865.0, 1.0), (60.0, 75.0, 4500.0, 1.0),
       (70.0, 57.0, 3990.0, 1.0), (58.0, 54.0, 3132.0, 1.0),
       (40.0, 34.0, 1360.0, 1.0), (61.0, 62.0, 3782.0, 1.0),
       (66.0, 50.0, 3300.0, 1.0), (37.0, 58.0, 2146.0, 1.0),
       (54.0, 48.0, 2592.0, 1.0), (77.0, 63.0, 4851.0, 1.0),
       (75.0, 74.0, 5550.0, 1.0), (57.0, 45.0, 2565.0, 1.0),
       (85.0, 71.0, 6035.0, 1.0), (82.0, 59.0, 4838.0, 1.0)], 
      dtype=[('x1', '<f8'), ('x3', '<f8'), ('x1*x3', '<f8'), ('1', '<f8')])
'''

from string import lowercase, uppercase

import numpy as np
from scipy.linalg import pinv

import sympy

from distutils.version import LooseVersion
SYMPY_0p6 = LooseVersion(sympy.__version__) < LooseVersion('0.7.0')

from nipy.fixes.sympy.utilities.lambdify import (implemented_function,
                                                 lambdify)

from nipy.algorithms.utils.matrices import matrix_rank, full_rank


def make_dummy(name):
    """ Make dummy variable of given name

    Parameters
    ----------
    name : str
        name of dummy variable

    Returns
    -------
    dum : `Dummy` instance

    Notes
    -----
    The interface to Dummy changed between 0.6.7 and 0.7.0
    """
    if SYMPY_0p6:
        return sympy.Symbol(name, dummy=True)
    return sympy.Dummy(name)


def define(*args, **kwargs):
    # Moved to utils module
    import warnings
    from . import utils
    warnings.warn('Please use define function from utils module',
                  DeprecationWarning,
                  stacklevel=2)
    return utils.define(*args, **kwargs)


class Term(sympy.Symbol):
    """A sympy.Symbol type to represent a term an a regression model

    Terms can be added to other sympy expressions with the single
    convention that a term plus itself returns itself.

    It is meant to emulate something on the right hand side of a formula
    in R. In particular, its name can be the name of a field in a
    recarray used to create a design matrix.

    >>> t = Term('x')
    >>> xval = np.array([(3,),(4,),(5,)], np.dtype([('x', np.float)]))
    >>> f = t.formula
    >>> d = f.design(xval)
    >>> print d.dtype.descr
    [('x', '<f8')]
    >>> f.design(xval, return_float=True)
    array([ 3.,  4.,  5.])
    """
    # This flag is defined to avoid using isinstance in getterms
    # and getparams.
    _term_flag = True

    def _getformula(self):
        return Formula([self])
    formula = property(_getformula, doc="Return a Formula with only terms=[self].")

    def __add__(self, other):
        if self == other:
            return self
        else:
            return sympy.Symbol.__add__(self, other)


# time symbol
T = Term('t')


def terms(names, **kwargs):
    ''' Return list of terms with names given by `names`

    This is just a convenience in defining a set of terms, and is the
    equivalent of ``sympy.symbols`` for defining symbols in sympy.

    We enforce the sympy 0.7.0 behavior of returning symbol "abc" from input
    "abc", rthan than 3 symbols "a", "b", "c".

    Parameters
    ----------
    names : str or sequence of str
       If a single str, can specify multiple ``Term``s with string
       containing space or ',' as separator.
    \\**kwargs : keyword arguments
       keyword arguments as for ``sympy.symbols``

    Returns
    -------
    ts : ``Term`` or tuple
       ``Term`` instance or list of ``Term`` instance objects named from `names`

    Examples
    --------
    >>> terms(('a', 'b', 'c'))
    (a, b, c)
    >>> terms('a, b, c')
    (a, b, c)
    >>> terms('abc')
    abc
    '''
    if (SYMPY_0p6
        and isinstance(names, basestring)
        and not set(', ').intersection(names)):
        if not kwargs.get('each_char', False):
            # remove each_char (or no-op if absent)
            kwargs.pop('each_char', None)
            names = (names,)
    syms = sympy.symbols(names, **kwargs)
    try:
        len(syms)
    except TypeError:
        return Term(syms.name)
    return tuple(Term(s.name) for s in syms)


class FactorTerm(Term):
    """ Boolean Term derived from a Factor.

    Its properties are the same as a Term except that its product with
    itself is itself.
    """
    # This flag is defined to avoid using isinstance in getterms
    _factor_term_flag = True

    def __new__(cls, name, level):
        new = Term.__new__(cls, "%s_%s" % (name, level))
        new.level = level
        new.factor_name = name
        return new

    def __mul__(self, other):

        if self == other:
            return self
        else:
            return sympy.Symbol.__mul__(self, other)


class Beta(sympy.symbol.Dummy):
    ''' A symbol tied to a Term `term` '''
    def __new__(cls, name, term):
        new = sympy.symbol.Dummy.__new__(cls, name)
        new._term = term
        return new


def getparams(expression):
    """ Return the parameters of an expression that are not Term 
    instances but are instances of sympy.Symbol.

    Examples
    --------
    >>> x, y, z = [Term(l) for l in 'xyz']
    >>> f = Formula([x,y,z])
    >>> getparams(f)
    []
    >>> f.mean
    _b0*x + _b1*y + _b2*z
    >>> getparams(f.mean)
    [_b0, _b1, _b2]
    >>> th = sympy.Symbol('theta')
    >>> f.mean*sympy.exp(th)
    (_b0*x + _b1*y + _b2*z)*exp(theta)
    >>> getparams(f.mean*sympy.exp(th))
    [theta, _b0, _b1, _b2]
    """
    atoms = set([])
    expression = np.array(expression)
    if expression.shape == ():
        expression = expression.reshape((1,))
    if expression.ndim > 1:
        expression = expression.reshape((np.product(expression.shape),))
    for term in expression:
        atoms = atoms.union(sympy.sympify(term).atoms())
    params = []
    for atom in atoms:
        if isinstance(atom, sympy.Symbol) and not is_term(atom):
            params.append(atom)
    params.sort()
    return params


def getterms(expression):
    """ Return the all instances of Term in an expression.

    Examples
    --------
    >>> x, y, z = [Term(l) for l in 'xyz']
    >>> f = Formula([x,y,z])
    >>> getterms(f)
    [x, y, z]
    >>> getterms(f.mean)
    [x, y, z]
    """
    atoms = set([])
    expression = np.array(expression)
    if expression.shape == ():
        expression = expression.reshape((1,))
    if expression.ndim > 1:
        expression = expression.reshape((np.product(expression.shape),))
    for e in expression:
        atoms = atoms.union(e.atoms())
    terms = []
    for atom in atoms:
        if is_term(atom):
            terms.append(atom)
    terms.sort()
    return terms


def make_recarray(rows, names, dtypes=None):
    """ Create recarray from `rows` with field names `names`

    Create a recarray with named columns from a list of rows and names
    for the columns. If dtype is None, the dtype is based on rows if it
    is an np.ndarray, else the data is cast as np.float. If dtypes are
    supplied, it uses the dtypes to create a np.dtype unless rows is an
    np.ndarray, in which case dtypes are ignored

    Parameters
    ----------
    rows: array-like
        Rows that will be turned into an recarray.
    names: sequence
        Sequence of strings - names for the columns.
    dtypes: [str or np.dtype]
        Used to create a np.dtype, can be np.dtypes or string.

    Returns
    -------
    v : np.ndarray

    Examples
    --------
    The following tests depend on machine byte order to pass

    >>> arr = np.array([[3,4 ], [4, 6], [6, 8]])
    >>> make_recarray(arr, ['x', 'y']) #doctest: +ELLIPSIS
    array([[(3, 4)],
           [(4, 6)],
           [(6, 8)]],
          dtype=[('x', '...'), ('y', '...')])
    >>> r = make_recarray(arr, ['w', 'u'])
    >>> make_recarray(r, ['x', 'y']) #doctest: +ELLIPSIS
    array([[(3, 4)],
           [(4, 6)],
           [(6, 8)]],
          dtype=[('x', '...'), ('y', '...')])
    >>> make_recarray([[3, 4], [4, 6], [7, 9]], 'wv', [np.float, np.int]) #doctest: +ELLIPSIS
    array([(3.0, 4), (4.0, 6), (7.0, 9)],
          dtype=[('w', '...'), ('v', '...')])
    """
    # XXX This function is sort of one of convenience
    # Would be nice to use DataArray or something like that
    # to add axis names.
    if isinstance(rows, np.ndarray):
        if rows.dtype.isbuiltin:
            dtype = np.dtype([(n, rows.dtype) for n in names])
        else:
            dtype = np.dtype([(n, d[1]) for n, d in zip(names, rows.dtype.descr)])
        if dtypes is not None:
            raise ValueError('dtypes not used if rows is an ndarray')
        return rows.view(dtype)
    if dtypes is None:
        dtype = np.dtype([(n, np.float) for n in names])
    else:
        dtype = np.dtype([(n, d) for n, d in zip(names, dtypes)])
    nrows = []
    vector = -1
    for r in rows:
        if vector < 0:
            a = np.array(r)
            if a.shape == ():
                vector = True
            else:
                vector = False
        if not vector:
            nrows.append(tuple(r))
        else:
            nrows.append(r)
    if vector:
        if len(names) != 1: # a 'row vector'
            nrows = tuple(nrows)
            return np.array(nrows, dtype)
        else:
            nrows = np.array([(r,) for r in nrows], dtype)
    return np.array(nrows, dtype)


class Formula(object):
    """ A Formula is a model for a mean in a regression model.

    It is often given by a sequence of sympy expressions, with the mean
    model being the sum of each term multiplied by a linear regression
    coefficient.

    The expressions may depend on additional Symbol instances,
    giving a non-linear regression model.
    """
    # This flag is defined to avoid using isinstance
    _formula_flag = True

    def __init__(self, seq, char = 'b'):
        """
        Parameters
        ----------
        seq : sequence of ``sympy.Basic``
        char : str, optional
            character for regression coefficient
        """
        self._terms = np.asarray(seq)
        self._counter = 0
        self.char = char

    # Properties
    def _getcoefs(self):
        if not hasattr(self, '_coefs'):
            self._coefs = {}
            for term in self.terms:
                self._coefs.setdefault(term, Beta("%s%d" % (self.char, self._counter), term))
                self._counter += 1
        return self._coefs
    coefs = property(_getcoefs, doc='Coefficients in the linear regression formula.')

    def _getterms(self):
        t = self._terms
        # The Rmode flag is meant to emulate R's implicit addition of an 
        # intercept to every formula. It currently cannot be changed.
        Rmode = False
        if Rmode:
            if sympy.Number(1) not in self._terms:
                t = np.array(list(t) + [sympy.Number(1)])
        return t
    terms = property(_getterms, doc='Terms in the linear regression formula.')

    def _getmean(self):
        """ Expression for mean

        Expression for the mean, expressed as a linear combination of
        terms, each with dummy variables in front.
        """
        b = [self.coefs[term] for term in self.terms]
        return np.sum(np.array(b)*self.terms)
    mean = property(_getmean, doc="Expression for the mean, expressed "
                    "as a linear combination of terms, each with dummy "
                    "variables in front.")

    def _getdiff(self):
        params = sorted(list(set(getparams(self.mean))))
        return [sympy.diff(self.mean, p).doit() for p in params]
    design_expr = property(_getdiff)

    def _getdtype(self):
        vnames = [str(s) for s in self.design_expr]
        return np.dtype([(n, np.float) for n in vnames])
    dtype = property(_getdtype, doc='The dtype of the design matrix of the Formula.')

    def __repr__(self):
        return """Formula(%s)""" % `list(self.terms)`

    def __getitem__(self, key):
        """ Return the term such that str(term) == key.

        Parameters
        ----------
        key : str
            name of term to retrieve

        Returns
        -------
        term : sympy.Expression
        """
        names = [str(t) for t in self.terms]
        try:
            idx = names.index(key)
        except ValueError:
            raise ValueError('term %s not found' % key)
        return self.terms[idx]

    @staticmethod
    def fromrec(rec, keep=[], drop=[]):
        """ Construct Formula from recarray

        For fields with a string-dtype, it is assumed that these are
        qualtiatitve regressors, i.e. Factors.

        Parameters
        ----------
        rec: recarray
            Recarray whose field names will be used to create a formula.
        keep: []
            Field names to explicitly keep, dropping all others.
        drop: []
            Field names to drop.
        """
        f = {}
        for n in rec.dtype.names:
            if rec[n].dtype.kind == 'S':
                f[n] = Factor.fromcol(rec[n], n)
            else:
                f[n] = Term(n).formula
        for d in drop:
            del(f[d])
        if keep:
            return np.sum([t for n, t in f.items() if n in keep])
        else:
            return np.sum(f.values())

    def subs(self, old, new):
        """ Perform a sympy substitution on all terms in the Formula

        Returns a new instance of the same class

        Parameters
        ----------
        old : sympy.Basic
           The expression to be changed
        new : sympy.Basic
           The value to change it to.

        Returns
        -------
        newf : Formula

        Examples
        --------
        >>> s, t = [Term(l) for l in 'st']
        >>> f, g = [sympy.Function(l) for l in 'fg']
        >>> form = Formula([f(t),g(s)])
        >>> newform = form.subs(g, sympy.Function('h'))
        >>> newform.terms
        array([f(t), h(s)], dtype=object)
        >>> form.terms
        array([f(t), g(s)], dtype=object)
        """
        return self.__class__([term.subs(old, new) for term in self.terms])

    def __add__(self, other):
        """
        Create a new Formula by combining terms
        of other with those of self.

        >>> x, y, z = [Term(l) for l in 'xyz']
        >>> f1 = Formula([x,y,z])
        >>> f2 = Formula([y])+I
        >>> f3=f1+f2
        >>> sorted(f1.terms)
        [x, y, z]
        >>> sorted(f2.terms)
        [1, y]
        >>> sorted(f3.terms)
        [1, x, y, y, z]
        """
        if not is_formula(other):
            raise ValueError('only Formula objects can be added to a Formula')
        f = self.__class__(np.hstack([self.terms, other.terms]))
        return f

    def __sub__(self, other):
        """
        Create a new Formula by deleting terms in other
        from self. No exceptions are raised for terms in other that do not appear in
        self.

        >>> x, y, z = [Term(l) for l in 'xyz']
        >>> f1 = Formula([x,y,z])
        >>> f2 = Formula([y])+I
        >>> f1.mean
        _b0*x + _b1*y + _b2*z
        >>> f2.mean
        _b0*y + _b1
        >>> f3=f2-f1
        >>> f3.mean
        _b0
        >>> f4=f1-f2
        >>> f4.mean
        _b0*x + _b1*z
        """
        if not is_formula(other):
            raise ValueError('only Formula objects can be subtracted from a Formula')
        d = list(set(self.terms).difference(other.terms))
        return self.__class__(d)

    def __array__(self):
        return self.terms

    def _getparams(self):
        return getparams(self.mean)
    params = property(_getparams, doc='The parameters in the Formula.')

    def __mul__(self, other):
        if not is_formula(other):
            raise ValueError('only two Formulas can be multiplied together')
        if is_factor(self):
            if self == other:
                return self
        v = []
        # Compute the pairwise product of each term
        # If either one is a Term, use Term's multiplication
        for sterm in self.terms:
            for oterm in other.terms:
                if is_term(sterm):
                    v.append(Term.__mul__(sterm, oterm))
                elif is_term(oterm):
                    v.append(Term.__mul__(oterm, sterm))
                else:
                    v.append(sterm*oterm)
        return Formula(tuple(np.unique(v)))

    def __eq__(self, other):
        s = np.array(self)
        o = np.array(other)
        if s.shape != o.shape:
            return False
        return np.alltrue(np.equal(np.array(self), np.array(other)))

    def _setup_design(self):
        """
        Create a callable object to evaluate the design matrix
        at a given set of parameter values to be specified by
        a recarray and observed Term values, also specified
        by a recarray.
        """
        # the design expression is the differentiation of the expression
        # for the mean.  It is a list
        d = self.design_expr
        # Before evaluating, we recreate the formula
        # with numbered terms, and numbered parameters.

        # This renaming has no impact on the
        # final design matrix as the
        # callable, self._f below, is a lambda
        # that does not care about the names of the terms.

        # First, find all terms in the mean expression,
        # and rename them in the form "__t%d__" with a
        # random offset.
        # This may cause a possible problem
        # when there are parameters named something like "__t%d__".
        # Using the random offset will minimize the possibility
        # of this happening.

        # This renaming is here principally because of the intercept.

        random_offset = np.random.random_integers(low=0, high=2**30)

        terms = getterms(self.mean)

        newterms = []
        for i, t in enumerate(terms):
            newt = sympy.Symbol("__t%d__" % (i + random_offset))
            for j, _ in enumerate(d):
                d[j] = d[j].subs(t, newt)
            newterms.append(newt)

        # Next, find all parameters that remain in the design expression.
        # In a standard regression model, there will be no parameters
        # because they will all be differentiated away in computing
        # self.design_expr. In nonlinear models, parameters will remain.

        params = getparams(self.design_expr)
        newparams = []
        for i, p in enumerate(params):
            newp = make_dummy("__p%d__" % (i + random_offset))
            for j, _ in enumerate(d):
                d[j] = d[j].subs(p, newp)
            newparams.append(newp)

        # If there are any aliased functions, these need to be added
        # to the name space before sympy lambdifies the expression

        # These "aliased" functions are used for things like
        # the natural splines, etc. You can represent natural splines
        # with sympy but the expression is pretty awful.  Note that
        # ``d`` here is list giving the differentiation of the
        # expression for the mean.  self._f(...) therefore also returns
        # a list
        self._f = lambdify(newparams + newterms, d, ("numpy"))

        # The input to self.design will be a recarray of that must 
        # have field names that the Formula will expect to see.
        # However, if any of self.terms are FactorTerms, then the field
        # in the recarray will not actually be in the Term.
        # 
        # For example, if there is a Factor 'f' with levels ['a','b'],
        # there will be terms 'f_a' and 'f_b', though the input to
        # design will have a field named 'f'. In this sense,
        # the recarray used in the call to self.design
        # is not really made up of terms, but "preterms".

        # In this case, the callable

        preterm = []
        for t in terms:
            if not is_factor_term(t):
                preterm.append(str(t))
            else:
                preterm.append(t.factor_name)
        preterm = list(set(preterm))

        # There is also an argument for parameters that are not
        # Terms. 

        self._dtypes = {'param':np.dtype([(str(p), np.float) for p in params]),
                        'term':np.dtype([(str(t), np.float) for t in terms]),
                        'preterm':np.dtype([(n, np.float) for n in preterm])}

        self.__terms = terms

    def design(self,
               input,
               param=None,
               return_float=False,
               contrasts=None):
        """ Construct the design matrix, and optional contrast matrices.

        Parameters
        ----------
        input : np.recarray
           Recarray including fields needed to compute the Terms in
           getparams(self.design_expr).
        param : None or np.recarray
           Recarray including fields that are not Terms in
           getparams(self.design_expr)
        return_float : bool, optional
           If True, return a np.float array rather than a np.recarray
        contrasts : None or dict, optional
           Contrasts. The items in this dictionary should be (str,
           Formula) pairs where a contrast matrix is constructed for
           each Formula by evaluating its design at the same parameters
           as self.design. If not None, then the return_float is set to True.

        Returns
        -------
        des : 2D array
            design matrix
        cmatrices : dict, optional
            Dictionary with keys from `contrasts` input, and contrast matrices
            corresponding to `des` design matrix.  Returned only if `contrasts`
            input is not None
        """
        self._setup_design()

        preterm_recarray = input
        param_recarray = param

        # The input to design should have field names for all fields in self._dtypes['preterm']
        if not set(preterm_recarray.dtype.names).issuperset(self._dtypes['preterm'].names):
            raise ValueError("for term, expecting a recarray with "
                             "dtype having the following names: %s"
                             % `self._dtypes['preterm'].names`)
        # The parameters should have field names for all fields in self._dtypes['param']
        if param_recarray is not None:
            if not set(param_recarray.dtype.names).issuperset(self._dtypes['param'].names):
                raise ValueError("for param, expecting a recarray with "
                                 "dtype having the following names: %s"
                                 % `self._dtypes['param'].names`)
        # If the only term is an intercept,
        # the return value is a matrix of 1's.
        if list(self.terms) == [sympy.Number(1)]:
            a = np.ones(preterm_recarray.shape[0], np.float)
            if not return_float:
                a = a.view(np.dtype([('intercept', np.float)]))
            return a
        elif not self._dtypes['term']:
            raise ValueError("none of the expresssions are self.terms "
                             "are Term instances; shape of resulting "
                             "undefined")
        # The term_recarray is essentially the same as preterm_recarray,
        # except that all factors in self are expanded
        # into their respective binary columns.
        term_recarray = np.zeros(preterm_recarray.shape[0], 
                                 dtype=self._dtypes['term'])
        for t in self.__terms:
            if not is_factor_term(t):
                term_recarray[t.name] = preterm_recarray[t.name]
            else:
                term_recarray['%s_%s' % (t.factor_name, t.level)] = \
                    np.array([x == t.level for x in preterm_recarray[t.factor_name]]).reshape(-1)
        # The lambda created in self._setup_design needs to take a tuple of
        # columns as argument, not an ndarray, so each column
        # is extracted and put into float_tuple.
        float_array = term_recarray.view(np.float)
        float_array.shape = (term_recarray.shape[0], -1)
        float_array = float_array.T
        float_tuple = tuple(float_array)
        # If there are any parameters, they also must be extracted
        # and put into a tuple with the order specified
        # by self._dtypes['param']
        if param_recarray is not None:
            param = tuple(float(param_recarray[n]) for n in self._dtypes['param'].names)
        else:
            param = ()
        # Evaluate the design at the parameters and tuple of arrays
        D = self._f(*(param+float_tuple))
        # TODO: check if this next stepis necessary
        # I think it is because the lambda evaluates sympy.Number(1) to 1
        # and not an array.
        D_tuple = [np.asarray(w) for w in D]

        need_to_modify_shape = []
        OK_row_shapes = []
        for i, row in enumerate(D_tuple):
            if row.shape in [(),(1,)]:
                need_to_modify_shape.append(i)
            else:
                OK_row_shapes.append(row.shape[0])
        # Make sure that each array has the correct shape.
        # The columns in need_to_modify should just be
        # the intercept column, which evaluates to have shape == ().
        # This makes sure that it has the correct number of rows.
        for i in need_to_modify_shape:
            D_tuple[i].shape = ()
            D_tuple[i] = np.multiply.outer(D_tuple[i], np.ones(preterm_recarray.shape[0]))
        # At this point, all the columns have the correct shape and the
        # design matrix is almost ready to output.
        D = np.array(D_tuple).T
        # If we will return a float matrix or any contrasts,
        # we may have some reshaping to do.
        if contrasts is None:
            contrasts = {}
        if return_float or contrasts:
            # If the design matrix is just a column of 1s
            # return a 1-dimensional array.
            D = np.squeeze(D.astype(np.float))
            # If there are contrasts, the pseudo-inverse of D
            # must be computed.
            if contrasts:
                if D.ndim == 1:
                    _D = D.reshape((D.shape[0], 1))
                else:
                    _D = D
                pinvD = np.linalg.pinv(_D)
        else:
            # Correct the dtype.
            # XXX There seems to be a lot of messing around with the dtype.
            # This would be a convenient place to just add
            # labels like a DataArray.
            D = np.array([tuple(r) for r in D], self.dtype)
        # Compute the contrast matrices, if any.
        if contrasts:
            cmatrices = {}
            for key, cf in contrasts.items():
                if not is_formula(cf):
                    cf = Formula([cf])
                L = cf.design(input, param=param_recarray, 
                              return_float=True)
                cmatrices[key] = contrast_from_cols_or_rows(L, _D, pseudo=pinvD)
            return D, cmatrices
        else:
            return D


def natural_spline(t, knots=None, order=3, intercept=False):
    """ Return a Formula containing a natural spline

    Spline for a Term with specified `knots` and `order`.

    Parameters
    ----------
    t : ``Term``
    knots : None or sequence, optional
       Sequence of float.  Default None (same as empty list)
    order : int, optional
       Order of the spline. Defaults to a cubic (==3)
    intercept : bool, optional
       If True, include a constant function in the natural
       spline. Default is False

    Returns
    -------
    formula : Formula
         A Formula with (len(knots) + order) Terms (if intercept=False,
         otherwise includes one more Term), made up of the natural spline
         functions.

    Examples
    --------
    >>> x = Term('x')
    >>> n = natural_spline(x, knots=[1,3,4], order=3)
    >>> xval = np.array([3,5,7.]).view(np.dtype([('x', np.float)]))
    >>> n.design(xval, return_float=True)
    array([[   3.,    9.,   27.,    8.,    0.,   -0.],
           [   5.,   25.,  125.,   64.,    8.,    1.],
           [   7.,   49.,  343.,  216.,   64.,   27.]])
    >>> d = n.design(xval)
    >>> print d.dtype.descr
    [('ns_1(x)', '<f8'), ('ns_2(x)', '<f8'), ('ns_3(x)', '<f8'), ('ns_4(x)', '<f8'), ('ns_5(x)', '<f8'), ('ns_6(x)', '<f8')]
    """
    if knots is None:
        knots = {}
    fns = []
    for i in range(order+1):
        n = 'ns_%d' % i
        def f(x, i=i):
            return x**i
        s = implemented_function(n, f)
        fns.append(s(t))

    for j, k in enumerate(knots):
        n = 'ns_%d' % (j+i+1,)
        def f(x, k=k, order=order):
            return (x-k)**order * np.greater(x, k)
        s = implemented_function(n, f)
        fns.append(s(t))

    if not intercept:
        fns.pop(0)

    ff = Formula(fns)
    return ff

# The intercept formula

I = Formula([sympy.Number(1)])


class Factor(Formula):
    """ A qualitative variable in a regression model

    A Factor is similar to R's factor. The levels of the Factor can be
    either strings or ints.
    """
    # This flag is defined to avoid using isinstance in getterms
    # and getparams.
    _factor_flag = True

    def __init__(self, name, levels, char='b'):
        """
        Parameters
        ----------
        name : str
        levels : [str or int]
            A sequence of strings or ints.
        char : str

        Returns
        -------
        """
        # Check whether they can all be cast to strings or ints without
        # loss.
        levelsarr = np.asarray(levels)
        if levelsarr.ndim == 0 and levelsarr.dtype.kind == 'S':
            levelsarr = np.asarray(list(levels))

        if levelsarr.dtype.kind != 'S': # the levels are not strings
            if not np.alltrue(np.equal(levelsarr, np.round(levelsarr))):
                raise ValueError('levels must be strings or ints')
            levelsarr = levelsarr.astype(np.int)

        Formula.__init__(self, [FactorTerm(name, l) for l in levelsarr], 
                        char=char)
        self.levels = list(levelsarr)
        self.name = name

    # TODO: allow different specifications of the contrasts
    # here.... this is like R's contr.sum

    def get_term(self, level):
        """
        Retrieve a term of the Factor...
        """
        if level not in self.levels:
            raise ValueError('level not found')
        return self["%s_%s" % (self.name, str(level))]

    def _getmaineffect(self, ref=-1):
        v = list(self._terms.copy())
        ref_term = v[ref]
        v.pop(ref)
        return Formula([vv - ref_term for vv in v])
    main_effect = property(_getmaineffect)

    def stratify(self, variable):
        """ Create a new variable, stratified by the levels of a Factor.

        Parameters
        ----------
        variable : str or a simple sympy expression whose string representation
            are all lower or upper case letters, i.e. it can be interpreted
            as a name

        Returns
        -------
        formula : Formula
            Formula whose mean has one parameter named variable%d, for each
            level in self.levels

        Examples
        --------
        >>> f = Factor('a', ['x','y'])
        >>> sf = f.stratify('theta')
        >>> sf.mean
        _theta0*a_x + _theta1*a_y
        """
        if not set(str(variable)).issubset(lowercase +
                                           uppercase + '0123456789'):
            raise ValueError('variable should be interpretable as a '
                             'name and not have anything but digits '
                             'and numbers')
        variable = sympy.sympify(variable)
        f = Formula(self._terms, char=variable)
        f.name = self.name
        return f

    @staticmethod
    def fromcol(col, name):
        """ Create a Factor from a column array.

        Parameters
        ----------
        col : ndarray
            an array with ndim==1

        name : str
            name of the Factor

        Returns
        -------

        factor : Factor

        Examples
        --------
        >>> data = np.array([(3,'a'),(4,'a'),(5,'b'),(3,'b')], np.dtype([('x', np.float), ('y', 'S1')]))
        >>> f1 = Factor.fromcol(data['y'], 'y')
        >>> f2 = Factor.fromcol(data['x'], 'x')
        >>> d = f1.design(data)
        >>> print d.dtype.descr
        [('y_a', '<f8'), ('y_b', '<f8')]
        >>> d = f2.design(data)
        >>> print d.dtype.descr
        [('x_3', '<f8'), ('x_4', '<f8'), ('x_5', '<f8')]
        """
        col = np.asarray(col)
        if col.ndim != 1 or (col.dtype.names and len(col.dtype.names) > 1):
            raise ValueError('expecting an array that can be thought '
                             'of as a column or field of a recarray')
        levels = np.unique(col)
        if not col.dtype.names and not name:
            name = 'factor'
        elif col.dtype.names:
            name = col.dtype.names[0]
        return Factor(name, levels)


def contrast_from_cols_or_rows(L, D, pseudo=None):
    """ Construct a contrast matrix from a design matrix D

    (possibly with its pseudo inverse already computed)
    and a matrix L that either specifies something in
    the column space of D or the row space of D.

    Parameters
    ----------
    L : ndarray
       Matrix used to try and construct a contrast.
    D : ndarray
       Design matrix used to create the contrast.
    pseudo : None or array-like, optional
       If not None, gives pseudo-inverse of `D`.  Allows you to pass
       this if it is already calculated.

    Returns
    -------
    C : ndarray
       Matrix with C.shape[1] == D.shape[1] representing an estimable
       contrast.

    Notes
    -----
    From an n x p design matrix D and a matrix L, tries to determine a p
    x q contrast matrix C which determines a contrast of full rank,
    i.e. the n x q matrix

    dot(transpose(C), pinv(D))

    is full rank.

    L must satisfy either L.shape[0] == n or L.shape[1] == p.

    If L.shape[0] == n, then L is thought of as representing
    columns in the column space of D.

    If L.shape[1] == p, then L is thought of as what is known
    as a contrast matrix. In this case, this function returns an estimable
    contrast corresponding to the dot(D, L.T)

    This always produces a meaningful contrast, not always
    with the intended properties because q is always non-zero unless
    L is identically 0. That is, it produces a contrast that spans
    the column space of L (after projection onto the column space of D).
    """
    L = np.asarray(L)
    D = np.asarray(D)
    n, p = D.shape
    if L.shape[0] != n and L.shape[1] != p:
        raise ValueError('shape of L and D mismatched')
    if pseudo is None:
        pseudo = pinv(D)
    if L.shape[0] == n:
        C = np.dot(pseudo, L).T
    else:
        C = np.dot(pseudo, np.dot(D, L.T)).T
    Lp = np.dot(D, C.T)
    if len(Lp.shape) == 1:
        Lp.shape = (n, 1)
    Lp_rank = matrix_rank(Lp)
    if Lp_rank != Lp.shape[1]:
        Lp = full_rank(Lp, Lp_rank)
        C = np.dot(pseudo, Lp).T
    return np.squeeze(C)


class RandomEffects(Formula):
    """ Covariance matrices for common random effects analyses.

    Examples
    --------
    Two subjects (here named 2 and 3):

    >>> subj = make_recarray([2,2,2,3,3], 's')
    >>> subj_factor = Factor('s', [2,3])

    By default the covariance matrix is symbolic.  The display differs a little
    between sympy versions (hence we don't check it in the doctests):

    >>> c = RandomEffects(subj_factor.terms)
    >>> c.cov(subj) #doctest: +IGNORE_OUTPUT
    array([[_s2_0, _s2_0, _s2_0, 0, 0],
           [_s2_0, _s2_0, _s2_0, 0, 0],
           [_s2_0, _s2_0, _s2_0, 0, 0],
           [0, 0, 0, _s2_1, _s2_1],
           [0, 0, 0, _s2_1, _s2_1]], dtype=object)

    With a numeric `sigma`, you get a numeric array:

    >>> c = RandomEffects(subj_factor.terms, sigma=np.array([[4,1],[1,6]]))
    >>> c.cov(subj)
    array([[ 4.,  4.,  4.,  1.,  1.],
           [ 4.,  4.,  4.,  1.,  1.],
           [ 4.,  4.,  4.,  1.,  1.],
           [ 1.,  1.,  1.,  6.,  6.],
           [ 1.,  1.,  1.,  6.,  6.]])
    """
    def __init__(self, seq, sigma=None, char = 'e'):
        """ Initialize random effects instance

        Parameters
        ----------
        seq : [``sympy.Basic``]
        sigma : ndarray
             Covariance of the random effects. Defaults
             to a diagonal with entries for each random
             effect.
        char : character for regression coefficient
        """

        self._terms = np.asarray(seq)
        q = self._terms.shape[0]

        self._counter = 0
        if sigma is None:
            self.sigma = np.diag([make_dummy('s2_%d' % i) for i in range(q)])
        else:
            self.sigma = sigma
        if self.sigma.shape != (q,q):
            raise ValueError('incorrect shape for covariance '
                             'of random effects, '
                             'should have shape %s' % repr(q,q))
        self.char = char

    def cov(self, term, param=None):
        """
        Compute the covariance matrix for some given data.

        Parameters
        ----------
        term : np.recarray
             Recarray including fields corresponding to the Terms in
             getparams(self.design_expr).
        param : np.recarray
             Recarray including fields that are not Terms in
             getparams(self.design_expr)

        Returns
        -------
        C : ndarray
             Covariance matrix implied by design and self.sigma.
        """
        D = self.design(term, param=param, return_float=True)
        return np.dot(D, np.dot(self.sigma, D.T))


def is_term(obj):
    """ Is obj a Term?
    """
    return hasattr(obj, "_term_flag")


def is_factor_term(obj):
    """ Is obj a FactorTerm?
    """
    return hasattr(obj, "_factor_term_flag")


def is_formula(obj):
    """ Is obj a Formula?
    """
    return hasattr(obj, "_formula_flag")


def is_factor(obj):
    """ Is obj a Factor?
    """
    return hasattr(obj, "_factor_flag")
