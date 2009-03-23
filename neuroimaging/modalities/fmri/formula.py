import sympy
import numpy as np

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
    for term in expression:
        atoms = atoms.union(term.atoms())

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

class Formula(object):
    
    """
    A Formula is a model for a mean in a regression model.

    It is often given by a sequence of sympy expressions,
    with the mean model being the sum of each term multiplied
    by a linear regression coefficient. 

    The expressions depend on additional Symbols,
    giving a non-linear regression model.

    """

    def __init__(self, seq, char = 'b', ignore=[]):
        """
        Inputs:
        -------
        seq : [``sympy.Basic``]

        char : character for regression coefficient

        ignore : [``sympy.Basic``]
             Ignore these symbols when differentiating
             to construct the design. 

        """

        self._aliases = {}
        self._terms = np.asarray(seq)
        self._counter = 0
        self.char = char
        self.ignore = ignore

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
        aliases = self.aliases.copy()
        if old in self.aliases.keys():
            aliases[new] = aliases[old]
            del(aliases[old])
        f = Formula([term.subs(old, new) for term in self.terms])
        for k, i in aliases.items():
            f.aliases[k] = i
        return f

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

    def _getaliases(self):
        return self._aliases
    aliases = property(_getaliases, doc="The aliases in the formula, to be used when the formula is lambdified.")

    def __add__(self, other):
        if not isinstance(other, Formula):
            raise ValueError('only Formula objects can be added together')
        f = Formula(np.hstack([self.terms, other.terms]))

        keys1 = self.aliases.keys()
        keys2 = other.aliases.keys()
        if set(keys1).intersection(keys2) != set([]):
            warnings.warn('two formulae have overlapping aliases')

        aliases = {}
        for key, value in self.aliases.items():
            f.aliases[key] = value
        for key, value in other.aliases.items():
            f.aliases[key] = value
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

    def _getdesign(self):
        p = list(set(getparams(self.mean)).difference(self.ignore))
        p.sort()
        return sympy.diff(self.mean, p)
    design = property(_getdesign, doc='Derivative of the mean function with respect to the linear and nonlinear parts of the Formula.')

    def _getdtype(self):
        vnames = [str(s) for s in self.design]
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

def natural_spline(t, knots=[], order=3, intercept=False):
    """
    Return a Formula containing a natural spline
    for a Term with specified knots and order.

    >>> x = Term('x')
    >>> n = natural_spline(x, knots=[1,3,4], order=3)
    >>> d = Design(n)
    >>> xval = np.array([3,5,7.]).view(np.dtype([('x', np.float)]))
    >>> d(xval)
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
    if not isinstance(t, Term):
        raise ValueError('expecting a Term')
    fns = []; symbols = []
    for i in range(order+1):
        n = 'ns_%d' % i
        s = sympy.Function(n)
        symbols.append(s(t))
        def anon(x,i=i):
            return x**i
        fns.append((n, anon))

    for j, k in enumerate(knots):
        n = 'ns_%d' % (j+i+1,)
        s = sympy.Function(n)
        symbols.append(s(t))
        def anon(x,k=k):
            return np.greater(x, k) * (x-k)**order
        fns.append((n, anon))

    if not intercept:
        fns.pop(0); symbols.pop(0)

    ff = Formula(symbols)
    for n, l in fns:
        ff.aliases[n] = l
    return ff

I = Formula([sympy.Number(1)])

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

class Design:

    """
    Used to evaluate the design matrix.
    """

    def _setup(self):
        """
        Create a callable object to evaluate the design matrix
        at a given set of parameter values and observed Term values.
        """
        d = self.formula.design

        # Before evaluating, we 'recreate' the formula
        # with numbered terms, and numbered parameters

        terms = getterms(self.formula.mean)
        newterms = []
        for i, t in enumerate(terms):
            newt = sympy.DeferredVector("t%d" % i)
            for j, _ in enumerate(d):
                d[j] = d[j].subs(t, newt)
            newterms.append(newt)

        params = getparams(self.formula.design)
        newparams = []
        for i, p in enumerate(params):
            newp = sympy.Symbol("p%d" % i, dummy=True)
            for j, _ in enumerate(d):
                d[j] = d[j].subs(p, newp)
            newparams.append(newp)

        self._f = sympy.lambdify(newparams + newterms, d, (self.formula._aliases, "numpy"))

        ptnames = []
        for t in terms:
            if not isinstance(t, FactorTerm):
                ptnames.append(str(t))
            else:
                ptnames.append(t.factor_name)
        ptnames = list(set(ptnames))

        self.dtypes = {'param':np.dtype([(str(p), np.float) for p in params]),
                        'term':np.dtype([(str(t), np.float) for t in terms]),
                        'preterm':np.dtype([(n, np.float) for n in ptnames])}
        self._terms = terms

    def __init__(self, formula, return_float=False):        

        self.formula = formula
        self._setup()
        self._return_float = return_float

    def __call__(self, preterm_recarray, param_recarray=None, return_float=False):
        if not set(preterm_recarray.dtype.names).issuperset(self.dtypes['preterm'].names):
            raise ValueError("for term, expecting a recarray with dtype having the following names: %s" % `self.dtypes['preterm'].names`)

        if param_recarray is not None:
            if not set(param_recarray.dtype.names).issuperset(self.dtypes['param'].names):
                raise ValueError("for param, expecting a recarray with dtype having the following names: %s" % `self.dtypes['param'].names`)

        term_recarray = np.zeros(preterm_recarray.shape[0], 
                                 dtype=self.dtypes['term'])
        for t in self._terms:
            if not isinstance(t, FactorTerm):
                term_recarray[t.name] = preterm_recarray[t.name]
            else:
                term_recarray['%s_%s' % (t.factor_name, t.level)] = \
                    np.array(map(lambda x: x == t.level, preterm_recarray[t.factor_name]))

        tnames = list(term_recarray.dtype.names)
        torder = [tnames.index(term) for term in self.dtypes['term'].names]

        float_array = term_recarray.view(np.float)
        float_array.shape = (term_recarray.shape[0], len(torder))
        float_tuple = tuple([float_array[:,i] for i in range(float_array.shape[1])])

        if param_recarray is not None:
            param = tuple(float(param_recarray[n]) for n in self.dtypes['param'].names)
        else:
            param = ()

        v = np.array(self._f(*(param+float_tuple))).T
        if return_float or self._return_float:
            return np.squeeze(v.astype(np.float))
        else:
            return np.array([tuple(r) for r in v], self.formula.dtype)

class Vectorize(Design):
    """
    This class can be used to take a (single-valued) sympy
    expression with only 't' as a Symbol and return a 
    callable that can be evaluated at an array of floats.

    Inputs:
    =======

    expr : sympy.Basic or Formula
        Expression with 't' the only Symbol. If it is a 
        Formula, then the only unknown symbol (besides 
        the coefficients) should be 't'.

    """

    def __init__(self, expr):
        if not isinstance(expr, Formula):
            expr = Formula([expr])
        Design.__init__(self, expr, return_float=True)

    def __call__(self, t):
        t = np.asarray(t).astype(np.float)
        tval = t.view(np.dtype([('t', np.float)]))
        return Design.__call__(self, tval)

###########################################################
        
"""
fMRI specific stuff
"""
t = sympy.Symbol('t')


# theta = sympy.Symbol('th')
# x = Term('x')
# y = Term('y')

# a = Formula([x])
# b = Formula([y])

# fac = Factor('f', 'ab')
# fac2 = Factor('f', 'ab')
# print fac.terms


# ff = (a + b)*fac + Formula([sympy.log(theta*x)])

# d = Design(ff)

# data = np.array([(3,4,'a'),(4,5,'b'), (5,6,'a')],
#                 dtype=np.dtype([('x', np.float),
#                                 ('y', np.float),
#                                 ('f', 'S1')]))
# param = np.array([(3.4,4.)], dtype=np.dtype([('th', np.float),
#                                              ('_b4', np.float)]))
# print d(data, param)

# f2 = (a+b + I)*fac + I
# dd = Design(f2)
# X = np.random.standard_normal((2000,2))

# data = np.zeros(2000, data.dtype)
# data['x'] = np.random.standard_normal((2000,))
# data['y'] = np.random.standard_normal((2000,))
# data['f'] = ['c']*10 + ['a']*990 + ['b']*1000
# Z =  dd(data, param)

# f3 = (a+b)*fac + I
