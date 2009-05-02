import sympy
import warnings
import numpy as np
from scipy.linalg import svdvals

from aliased import aliased_function, _add_aliases_to_namespace, vectorize

class Term(sympy.Symbol):

    """
    A Term is a term in a linear regression model. Terms can be added
    to other sympy expressions with the single convention that a 
    term plus itself returns itself.

    """

    def _getformula(self):
        return Formula([self])
    formula = property(_getformula, doc="Return a Formula with only terms=[self].")

    def __add__(self, other):
        if self == other:
            return self
        else:
            return sympy.Symbol.__add__(self, other)

class FactorTerm(Term):
    """
    Boolean Term derived from a Factor.
    """

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

    def __new__(cls, name, term):
        new = sympy.symbol.Dummy.__new__(cls, name)
        new._term = term
        return new
        
def getparams(expression):
    """
    Return the parameters of an expression that are not Term 
    but are sympy.Symbols.
    """
    atoms = set([])
    expression = np.array(expression)
    if expression.shape == ():
        expression.shape = (1,)
    if expression.ndim > 1:
        expression.shape = np.product(expression.shape)
    for term in expression:
        atoms = atoms.union(sympy.sympify(term).atoms())

    params = []
    for atom in atoms:
        if not isinstance(atom, Term) and isinstance(atom, sympy.Symbol):
            params.append(atom)
    params.sort()
    return params

def getterms(expression):
    """
    Return the Terms of an expression.
    """
    atoms = set([])
    expression = np.array(expression)
    if expression.shape == ():
        expression.shape = (1,)
    for term in expression:
        atoms = atoms.union(term.atoms())

    terms = []
    for atom in atoms:
        if isinstance(atom, Term):
            terms.append(atom)
    terms.sort()
    return terms

def make_recarray(rows, names, dtypes=None):
    """
    Create a recarray with named column
    from a list of rows and names for the
    columns. If dtype is None,
    the dtype is based on rows if it
    is an np.ndarray, else
    the data is cast as np.float. If dtypes
    are supplied,
    it uses the dtypes to create a np.dtype
    unless rows is an np.ndarray, in which
    case dtypes are ignored

    Parameters:
    -----------

    rows: []
        Rows that will be turned into an array.

    names: [str]
        Names for the columns.

    dtypes: [str or np.dtype]
        Used to create a np.dtype, can be np.dtypes or string.

    Outputs:
    --------

    v : np.ndarray

    >>> arr = np.array([[3,4],[4,6],[6,8]])
    >>> make_recarray(arr, ['x','y'])
    array([[(3, 4)],
           [(4, 6)],
           [(6, 8)]], 
          dtype=[('x', '<i4'), ('y', '<i4')])
    >>> r = make_recarray(arr, ['w', 'u'])
    >>> make_recarray(r, ['x','y'])
    array([[(3, 4)],
           [(4, 6)],
           [(6, 8)]], 
          dtype=[('x', '<i4'), ('y', '<i4')])

    >>> make_recarray([[3,4],[4,6],[7,9]], 'wv', [np.float, np.int])
    array([(3.0, 4), (4.0, 6), (7.0, 9)], 
          dtype=[('w', '<f8'), ('v', '<i4')])
    >>> 

    """

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
    
    """
    A Formula is a model for a mean in a regression model.

    It is often given by a sequence of sympy expressions,
    with the mean model being the sum of each term multiplied
    by a linear regression coefficient. 

    The expressions depend on additional Symbols,
    giving a non-linear regression model.

    """

    def __init__(self, seq, char = 'b'):
        """
        Inputs:
        -------
        seq : [``sympy.Basic``]

        char : character for regression coefficient


        """

        self._terms = np.asarray(seq)
        self._counter = 0
        self.char = char

    def subs(self, old, new):
        """
        Perform a sympy substitution on all terms in the Formula,
        returning a new Formula.

        Inputs:
        =======

        old : sympy.Basic
            The expression to be changed

        new : sympy.Basic
            The value to change it to.
        
        Outputs: 
        ========
        
        newf : Formula

        
        >>> s, t= [Term(l) for l in 'st']
        >>> f, g = [sympy.Function(l) for l in 'sg']
        >>> ff = Formula([f(t),f(s)])
        >>> ff.mean
        _b0*f(t) + _b1*f(s)
        >>> gf = ff.subs(f,g)
        >>> gf.mean
        _b0*g(t) + _b1*g(s)

        """
        return Formula([term.subs(old, new) for term in self.terms])

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
        Rmode = False
        if Rmode:
            if sympy.Number(1) not in self._terms:
                t = np.array(list(t) + [sympy.Number(1)])
        return t
    terms = property(_getterms, doc='Terms in the linear regression formula.')

    def __add__(self, other):
        if not isinstance(other, Formula):
            raise ValueError('only Formula objects can be added together')
        f = Formula(np.hstack([self.terms, other.terms]))
        return f

    def __array__(self):
        return self.terms

    def _getparams(self):
        return getparams(self.mean)
    params = property(_getparams, doc='The parameters in the Formula.')

    def _getmean(self):
        """
        Expression for the mean, expressed as a linear
        combination of terms, each with dummy variables in front.
        """
        b = [self.coefs[term] for term in self.terms]
        return np.sum(np.array(b)*self.terms)

    mean = property(_getmean, doc="Expression for the mean, expressed as a linear combination of terms, each with dummy variables in front.")

    def _getdiff(self):
        p = list(set(getparams(self.mean)))
        p.sort()
        return sympy.diff(self.mean, p)
    design_expr = property(_getdiff)

    def _getdtype(self):
        vnames = [str(s) for s in self.design_expr]
        return np.dtype([(n, np.float) for n in vnames])
    dtype = property(_getdtype, doc='The dtype of the design matrix of the Formula.')

    def __mul__(self, other):
        if not hasattr(other, 'terms'):
            raise ValueError('must have terms to be multiplied')

        if isinstance(self, Factor):
            if self == other:
                return self

        v = []
        for sterm in self.terms:
            for oterm in other.terms:
                if isinstance(sterm, Term):
                    v.append(Term.__mul__(sterm, oterm))
                elif isinstance(oterm, Term):
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
        at a given set of parameter values and observed Term values.
        """
        d = self.design_expr

        # Before evaluating, we 'recreate' the formula
        # with numbered terms, and numbered parameters

        terms = getterms(self.mean)
        newterms = []
        for i, t in enumerate(terms):
            newt = sympy.DeferredVector("t%d" % i)
            for j, _ in enumerate(d):
                d[j] = d[j].subs(t, newt)
            newterms.append(newt)

        params = getparams(self.design_expr)
        newparams = []
        for i, p in enumerate(params):
            newp = sympy.Symbol("p%d" % i, dummy=True)
            for j, _ in enumerate(d):
                d[j] = d[j].subs(p, newp)
            newparams.append(newp)


        self.n = {}; 
        for _d in d:
            _add_aliases_to_namespace(_d, self.n)

        #TODO: use aliased.lambdify for this?
        self._f = sympy.lambdify(newparams + newterms, d, (self.n, "numpy"))
        ptnames = []
        for t in terms:
            if not isinstance(t, FactorTerm):
                ptnames.append(str(t))
            else:
                ptnames.append(t.factor_name)
        ptnames = list(set(ptnames))

        self.dtypes = {'param':np.dtype([(str(p), np.float) for p in params]),
                        'term':np.dtype([(str(t), np.float) for t in terms]),
                        'preterm':np.dtype([(na, np.float) for na in ptnames])}
        self.__terms = terms

    def design(self, term, param=None, return_float=False,
               contrasts={}):
        """
        Construct the design matrix, and optional
        contrast matrices.

        Parameters:
        -----------

        term : np.recarray
             Recarray including fields corresponding to the Terms in 
             getparams(self.design_expr).

        param : np.recarray
             Recarray including fields that are not Terms in 
             getparams(self.design_expr)
        
        return_float : bool
             Return a np.float array or an np.recarray

        contrasts : {}
             Contrasts. The items in this dictionary
             should be (str, Formula) pairs where
             a contrast matrix is constructed for each Formula
             by evaluating its design at the same parameters as self.design.

        """
        self._setup_design()

        preterm_recarray = term
        param_recarray = param

        if not set(preterm_recarray.dtype.names).issuperset(self.dtypes['preterm'].names):
            raise ValueError("for term, expecting a recarray with dtype having the following names: %s" % `self.dtypes['preterm'].names`)

        if param_recarray is not None:
            if not set(param_recarray.dtype.names).issuperset(self.dtypes['param'].names):
                raise ValueError("for param, expecting a recarray with dtype having the following names: %s" % `self.dtypes['param'].names`)

        term_recarray = np.zeros(preterm_recarray.shape[0], 
                                 dtype=self.dtypes['term'])
        for t in self.__terms:
            if not isinstance(t, FactorTerm):
                term_recarray[t.name] = preterm_recarray[t.name]
            else:
                term_recarray['%s_%s' % (t.factor_name, t.level)] = \
                    np.array(map(lambda x: x == t.level, preterm_recarray[t.factor_name]))

        tnames = list(term_recarray.dtype.names)
        torder = [tnames.index(_term) for _term in self.dtypes['term'].names]

        float_array = term_recarray.view(np.float)
        float_array.shape = (term_recarray.shape[0], len(torder))
        float_tuple = tuple([float_array[:,i] for i in range(float_array.shape[1])])

        if param_recarray is not None:
            param = tuple(float(param_recarray[n]) for n in self.dtypes['param'].names)
        else:
            param = ()

        v = self._f(*(param+float_tuple))
        varr = [np.array(w) for w in v]
        
        m = []
        l = []
        for i, w in enumerate(varr):
            if w.shape in [(),(1,)]:
                m.append(i)
            else:
                l.append(w.shape[0])
        if not np.alltrue(np.equal(l, l[0])):
            raise ValueError, 'shape mismatch'

        # Multiply all numbers by columns of 1s

        for i in m:
            varr[i] = varr[i] * np.ones(l[0])

        v = np.array(varr).T
        if return_float or contrasts:
            D = np.squeeze(v.astype(np.float))
            if contrasts:
                pinvD = np.linalg.pinv(D)
        else:
            D = np.array([tuple(r) for r in v], self.dtype)

        cmatrices = {}
        for key, cf in contrasts.items():
            if not isinstance(cf, Formula):
                cf = Formula([cf])
            L = cf.design(term, param=param_recarray, 
                          return_float=True)
            cmatrices[key] = contrast_from_cols_or_rows(L, D, pseudo=pinvD)
            
        if not contrasts:
            return D
        return D, cmatrices

def natural_spline(t, knots=[], order=3, intercept=False):
    """
    Return a Formula containing a natural spline
    for a Term with specified knots and order.

    >>> x = Term('x')
    >>> n = natural_spline(x, knots=[1,3,4], order=3)
    >>> xval = np.array([3,5,7.]).view(np.dtype([('x', np.float)]))
    >>> n.design(xval)
    array([(3.0, 9.0, 27.0, 8.0, 0.0, -0.0),
           (5.0, 25.0, 125.0, 64.0, 8.0, 1.0),
           (7.0, 49.0, 343.0, 216.0, 64.0, 27.0)],
          dtype=[('ns_1(x)', '<f8'), ('ns_2(x)', '<f8'), ('ns_3(x)', '<f8'), ('ns_4(x)', '<f8'), ('ns_5(x)', '<f8'), ('ns_6(x)', '<f8')])
    >>>
                    
    Inputs:
    =======
    t : Term

    knots : [float]

    order : int
         Order of the spline. Defaults to a cubic.

    intercept : bool
         If True, include a constant function in the natural spline.

    Outputs:
    --------
    formula : Formula
         A Formula with (len(knots) + order) Terms
         (if intercept=False, otherwise includes one more Term), 
         made up of the natural spline functions.

    """
    fns = []
    for i in range(order+1):
        n = 'ns_%d' % i
        def f(x, i=i):
            return x**i
        s = aliased_function(n, f)
        fns.append(s(t))

    for j, k in enumerate(knots):
        n = 'ns_%d' % (j+i+1,)
        def f(x, k=k, order=order):
            return (x-k)**order * np.greater(x, k)
        s = aliased_function(n, f)
        fns.append(s(t))

    if not intercept:
        fns.pop(0)

    ff = Formula(fns)
    return ff

I = Formula([sympy.Number(1)])

# def factor(name, levels):
#     """
#     Experimenting with a different way of writing Factors...
#     """
#     t = Term(name)
#     fs = []
#     for l in levels:
#         def f(x, l=l):
#             return np.array([xx == l for xx in x])
#         fs.append(aliased_function('ind_%s' % (str(l)), f)(t))
#     return Formula(fs)


class Factor(Formula):

    def __init__(self, name, levels, char='b'):
        Formula.__init__(self, [FactorTerm(name, l) for l in levels], 
                        char=char)
        self.name = name

    # TODO: allow different specifications of the contrasts
    # here.... this is like R's contr.sum

    def _getmaineffect(self, ref=-1):
        v = list(self._terms.copy())
        ref_term = v[ref]
        v.pop(ref)
        return Formula([vv - ref_term for vv in v])
    main_effect = property(_getmaineffect)

    def stratify(self, varname):
        """
        Create a new variable, stratified by the levels of a Factor.

        :Inputs:
        --------

        varname : str
        """
        f = Formula(self._terms, char=varname)
        f.name = self.name
        return f

def contrast_from_cols_or_rows(L, D, pseudo=None):
    """
    Construct a contrast matrix from a design matrix D
    (possibly with its pseudo inverse already computed)
    and a matrix L that either specifies something in
    the column space of D or the row space of D.

    Parameters:
    -----------

    L : ndarray
         Matrix used to try and construct a contrast.

    D : ndarray
         Design matrix used to create the contrast.

    Outputs:
    --------

    C : ndarray
         Matrix with C.shape[1] == D.shape[1] representing
         an estimable contrast.

    Notes:
    ------

    From an n x p design matrix D and a matrix L, tries
    to determine a p x q contrast matrix C which
    determines a contrast of full rank, i.e. the
    n x q matrix

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
        raise ValueError, 'shape of L and D mismatched'

    if pseudo is None:
        pseudo = pinv(D)

    if L.shape[0] == n:
        C = np.dot(pseudo, L).T
    else:
        C = L
        C = np.dot(pseudo, np.dot(D, C.T)).T
        
    Lp = np.dot(D, C.T)

    if len(Lp.shape) == 1:
        Lp.shape = (n, 1)
        
    if rank(Lp) != Lp.shape[1]:
        Lp = fullrank(Lp)
        C = np.dot(pseudo, Lp).T

    return np.squeeze(C)

def rank(X, cond=1.0e-12):
    """
    Return the rank of a matrix X based on its generalized inverse,
    not the SVD.
    """
    X = np.asarray(X)
    if len(X.shape) == 2:
        D = svdvals(X)
        return int(np.add.reduce(np.greater(D / D.max(), cond).astype(np.int32)))
    else:
        return int(not np.alltrue(np.equal(X, 0.)))

def fullrank(X, r=None):
    """
    Return a matrix whose column span is the same as X.

    If the rank of X is known it can be specified as r -- no check
    is made to ensure that this really is the rank of X.

    """

    if r is None:
        r = rank(X)

    V, D, U = L.svd(X, full_matrices=0)
    order = np.argsort(D)
    order = order[::-1]
    value = []
    for i in range(r):
        value.append(V[:,order[i]])
    return np.asarray(np.transpose(value)).astype(np.float64)

class RandomEffects(Formula):
    """
    This class can be used to 
    construct covariance matrices for common
    random effects analyses.

    >>> subj = make_recarray([2,2,2,3,3], 's')
    >>> subj_factor = Factor('s', [2,3])
    >>> c = RandomEffects(subj_factor.terms)
    >>> c.cov(s)
    array([[_s2_0, _s2_0, _s2_0, 0, 0],
           [_s2_0, _s2_0, _s2_0, 0, 0],
           [_s2_0, _s2_0, _s2_0, 0, 0],
           [0, 0, 0, _s2_1, _s2_1],
           [0, 0, 0, _s2_1, _s2_1]], dtype=object)
    >>> c = RandomEffects(subj_factor.terms, sigma=np.array([[4,1],[1,6]]))
    >>> c.cov(s)
    array([[ 4.,  4.,  4.,  1.,  1.],
           [ 4.,  4.,  4.,  1.,  1.],
           [ 4.,  4.,  4.,  1.,  1.],
           [ 1.,  1.,  1.,  6.,  6.],
           [ 1.,  1.,  1.,  6.,  6.]])
    >>> 

    """
    def __init__(self, seq, sigma=None, char = 'e'):
        """
        Inputs:
        -------
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
            self.sigma = np.diag([sympy.Symbol('s2_%d' % i, dummy=True) for i in 
                                  range(q)])
        else:
            self.sigma = sigma
        if self.sigma.shape != (q,q):
            raise ValueError('incorrect shape for covariance of random effects, should have shape %s' % `(q,q)`)
        self.char = char

    def cov(self, term, param=None):
        """
        Compute the covariance matrix for
        some given data.

        Parameters:
        -----------

        term : np.recarray
             Recarray including fields corresponding to the Terms in 
             getparams(self.design_expr).

        param : np.recarray
             Recarray including fields that are not Terms in 
             getparams(self.design_expr)
        
        Outputs:
        --------

        C : ndarray
             Covariance matrix implied by design and self.sigma.

        """
        D = self.design(term, param=param, return_float=True)
        return np.dot(D, np.dot(self.sigma, D.T))

def define(name, expr):
    """
    Take an expression of 't' (possibly complicated)
    and make it a '%s(t)' % name, such that
    when it evaluates it has the right values.

    Parameters:
    -----------

    expr : sympy expression, with only 't' as a Symbol

    name : str

    Outputs:
    --------

    nexpr: sympy expression

    >>> t = Term('t')
    >>> expr = t**2 + 3*t
    >>> print expr
    3*t + t**2
    >>> newexpr = define('f', expr)
    >>> print newexpr
    f(t)
    >>> import aliased
    >>> f = aliased.lambdify(t, newexpr)
    >>> f(4)
    28
    >>> 3*4+4**2
    28
    >>> 

    """
    v = vectorize(expr)
    return aliased_function(name, v)(Term('t'))

