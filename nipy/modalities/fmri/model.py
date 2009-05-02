import numpy as np
import sympy as sym
from string import join

from utils import events,linear_interp, blocks
from formula import Term, Formula, Design, natural_spline
import hrf
from odict import OrderedDict

aliases = {'glover': hrf.glover, 'dglover':hrf.dglover, 'iglover':hrf.iglover}

def _string_from_term(term):
    """
    create a string that defines the name of a term
    """
    if hasattr(term,'name'):
        lst = term.name
    else:
        lst = []
        for t in term.args:
            lst.append(_string_from_term(t))
        lst=join(lst)
    return lst


def _contrast(cont, terms):
    """
    return the index of the terms that are involved in cont
    and the associated coefficient
    INPUT:
    - cont: a forumla specifying a contrast
    - terms: a list of design terms in which the coefficient mah be found
    OUPUT:
    -coeffs: the coefficients of the linear combination involved in the contrast
    - terms: the corresponding terms index
    """
    coeffs = []
    indices = []
    if hasattr(cont,'name'):
        w = [cont==t for t in terms]
        wf = filter(None, w)
        if len(wf):
            coeffs += [int(wf[0])]
            indices += [w.index(wf[0])]
        return coeffs,indices
                
    args = cont.args
    for a in args:
        w = [a.as_coefficient(t) for t in terms]
        wf = filter(None, w)
        if len(wf):
            coeffs += [int(wf[0])]
            indices += [w.index(wf[0])]
    return coeffs, indices




class LinearModel:
    
    def __init__(self, hrf):
        """
        Init th elinear models by creating ordered dictionaries
        that include conditions and regressors
        Also sepcify an hrf model

        INPUT:
        hrf(string or list of strings): the hrf model.
        So far, to be chosen within 'glover' and 'dglover' 

        CLASS MEMBERS
        - self.hrf: a list of hrf models
        
        TODO : each term (condition/regressor)  should become a class
        """
        if isinstance(hrf,list):
            self.hrf = hrf
        else:
            self.hrf = [hrf]
        self._conditions = OrderedDict()#{}
        self._extra_regressors = OrderedDict()#{}
        
    def add_condition(self, term, onsets, amplitudes=None, durations=None,hrf=None):
        """
        specify a new condition from the onset, amplitudes and duration values
        a specific hrf model can also be attatched to the condition.
        the conditions are appended to self.conditions

        INPUT:
        - term: the symbol that defines the condition
        - onsets: the onset times of  the events/blocks
        - amplitudes=None: the amplitudes values associated
        with the evnts/blocks.  
        By default these are equal to 1
        - durations=None: duration of the blocks.
        By default it is 0 and the trials are handled as  events
        - hrf=None: by default, the hrf of the model is used
        
        FIXME:
        - blocks are trated correctly, but the model is awkward
        - add modulations
        """
        if hrf==None:
            hrf = self.hrf 
            hrf_sym = [sym.Function(h) for h in self.hrf]
        else:
            hrf_sym = [sym.Function(h) for h in hrf]
        
        if durations==None:
            ## it is a problem to index _conditions by terms ?
            sft = _string_from_term(term)
            formula = Formula([events(onsets, amplitudes=amplitudes, f=f) for f in hrf_sym])
            self._conditions[term] = {'formula':formula, 'hrf model':hrf, 'id':sft}        
        else:
            # for blocks (bad and temporary)
            tstart = onsets
            nb = np.size(onsets)
            tend = onsets+durations
            if amplitudes==None:
                amp = np.ones(nb)
            else:
                amp = amplitudes

            cp = np.sort(np.hstack((tstart,tend)))
            cv = np.ravel(np.vstack((amp,-np.ones(nb))).T)
            sft = _string_from_term(term)

            lhrf = []
            
            for f in hrf_sym:
                if f not in[sym.Function('glover'),sym.Function('dglover')]:
                    raise ValueError, 'not implemented for blocks'
                
                """
                intervals = [[ts,te] for ts,te in zip(tstart,tend)]
                formula = Formula(blocks(intervals,amp)) 
                """                
                if f==sym.Function('glover'):
                    lhrf.append(sym.Function('iglover'))
                else:
                    lhrf.append(sym.Function('glover'))

            formula=Formula([events(cp,amplitudes=cv,f=f) for f in lhrf])
            self._conditions[term] = {'formula':formula, 'hrf model':hrf, 'id':sft}
            

    def regressor_names(self):
        """
        tentative function to get the names of the regressors
        """
        regnames = []
        for term in self._conditions:
            cname = self._conditions[term]['id']
            hrf_model = self._conditions[term]['hrf model']
            if isinstance(hrf_model,str):
                 regnames.append(join((cname,':',hrf_model)))
            else:
                for f in hrf_model:
                    regnames.append(join((cname,':',f)))
        for term in self._extra_regressors:
            cname = self._extra_regressors[term]['id']
            order = self._extra_regressors[term]['order']
            for o in range(order):
                regnames.append(join((cname,':',str(o))))
        return regnames

    def add_regressor(self,term,values,timestamps,order=1): 
        """
        Add a regressor as a time function interporalted from values
        sampled at timestamps
        INPUT:
        term: symbol to be added as regressor
        values: arrau of shape (n): the provided values
        timestamps: array of shape (n): the time stamps associated with the values
        order=1 : order of the spline model
        TODO:
        add hrf model 
        add multi-dimensional terms
        """
        t = Term('t')
        if order==1:
            formula = linear_interp(timestamps, values)
        sft = _string_from_term(term)
        self._extra_regressors[term]={'formula':formula, 'id':sft,'hrf_model':'None','order':1}
        return

    def _add_baseline(self):
        """
        add the baseline term in the regressors list
        """
        t = Term('t')
        aux = 1+t*1.e-30
        formula = Formula(aux)
        self._extra_regressors[Term('baseline')]={'formula':formula, 'id':'baseline','order':1}
        
    def polynomial_drift(self,order):
        """
        create the drift terms as polynomials
        these are added in self.extra_regressors
        INPUT:
        order (int): polynom order
        """
        t = Term('t')
        aux = [t**(i+1) for i in range(order)]
        formula = Formula(aux)
        self._extra_regressors[Term('polynomial_drift')]={'formula':formula, 'order':order,'id':'polynomial drift'}
        
            
    def cosine_drift(self,duration,hfcut=128):
        """
        Create the dirft terms as cosine function of time
        These are added in self.extra_regressors

        INPUT:
        - duration: duration of the experiments in seconds
        - hfcut=128: frequency cut in seconds
        """
        t = Term('t')
        # create the DCT basis
        numreg = int(np.floor(2 * float(duration) / float(hfcut)) + 1)
        aux = [sym.sqrt(2.0/duration) * sym.cos(sym.pi*(t/duration)*i) for i in range(1,numreg)]
        formula = Formula(aux)
        self._extra_regressors[Term('cosine_drift')]={'formula':formula, 'order':numreg,'id':'cosine drift'}
        
    def _drift(self, order, expression=None):
        """
        Create the drift terms
        seems to be consistent only for polynomial drifts
        warning:deprecated
        """
        t = Term('t')
        if isinstance(expression, sym.function.FunctionClass): 
            monom = expression(t)
        else: 
            monom = t
        aux = [monom**(i+1) for i in range(order)]
        self._extra_regressors[Term('drift')] = Formula(aux)

    
    def conditions(self):
        """
        return the terms correponding to the conditions of the model
        """
        conditions = self._conditions.keys()
        conditions.sort()
        return conditions
        
    
    def extra_regressors(self):
        """
        return the 'extra regressors' of the model
        """
        return [term for term in self._extra_regressors]

    def terms(self):
        """
        return all the terms of the models
        """
        return self.conditions()+ self.extra_regressors() 
 
    def formula(self):
        """
        defines the formula of the model
        """
        f = Formula([])
        cimap = np.zeros(len(self._conditions)+len(self._conditions))
        q = 0
        for term in self._conditions:
            f += self._conditions[term]['formula']
            cimap[q+1:]+=len(self._conditions[term]['hrf model'])
            q+=1
            
        for term in self._extra_regressors:
            f += self._extra_regressors[term]['formula']
            cimap[q+1:]+=self._extra_regressors[term]['order']
            q+=1

        self.cimap = cimap
        # setting the aliases
        for h in self.hrf: 
            f.aliases[h] = aliases[h]
        f.aliases['iglover']=aliases['iglover']
        return f

    def design_matrix(self, timestamps,addbaseline=True): 
        """
        X = self.design_matrix(timestamps)
        create a design matrix from the forula
        
        INPUT
        timestamps is a 1d array. 
        addbaseline=True: if True, add a  constant (baseline) term
        """
        if addbaseline:
             # add a constant baseline
             self._add_baseline()
             
        tval = np.asarray(timestamps, dtype=np.float).view(np.dtype([('t', np.float)]))
        D = Design(self.formula(), return_float=True)
        X = D(tval)
        nreg = X.shape[1]
        
        # get the permutation performed during the Design creation (sic!)
        # and invert it to retrieve the columns in a standard order
        order =  D.formula.params
        perm = [int(order[k].name[1:]) for k in range(nreg)]

        # compute the inverse permutation
        iperm = np.arange(nreg)
        iperm[perm]=np.arange(nreg)
        X = X[:,iperm]

        return X

    def _contrast_(self, cont):
        """
        C = self.contrast(cont)

        cont is a linear combination of terms (a symbolic expression).
        Return a matrix pxq where p is the number of columns of the
        design matrix and q is the 'dimensionality' of the contrast.
        DEPRECATED
        """
        nregressors = len(self.formula().terms)
        nhrfs = len(self.hrf)
        mat = np.zeros([nregressors, nhrfs])

        # Try contrast on conditions
        coeffs, indices = _contrast(cont, self.conditions())        
        if len(coeffs): 
            J = np.arange(nhrfs)
            for (c,i) in zip(coeffs, indices):
                mat[nhrfs*i+J, J] = c
        
        # Otherwise try contrast on extra regressors
        else:
            coeffs, indices = _contrast(cont, self.extra_regressors()) 
            if len(coeffs):       
                j = nhrfs*len(self.conditions())
                for (c,i) in zip(coeffs, indices):
                    mat[i+j, j] = c
        return mat

            
    def contrast(self, cont,hrf_term=None,verbose=0):
        """
        C = self.contrast(cont)

        INPUT:
        cont is a linear combination of terms (a symbolic expression).
        hrf_term: list or string is a specification  of the hrf
        Return a matrix pxq where p is the number of columns of the
        design matrix and q is the 'dimensionality' of the contrast.

        TODO:  introduce the concept of restriction
        e.g. : 'contrast c1-c2 restricted to the glover derivative'
        """
        # todo : check that the formula has been computed before
        nregressors = len(self.formula().terms)
        
        # first get the terms corresponding to the 
        coeffs, indices = _contrast(cont, self.conditions())
        if verbose:
            print "term coeffs",coeffs
            print "term indices",indices

        # if there are no condition involved,
        # try contrast on extra regressors
        if len(coeffs)==0:
            print cont
            coeffs, indices = _contrast(cont, self.extra_regressors()) 
            mat = np.zeros((nregressors,1))
            # fixme : the dimension should be arbitrary
            if len(coeffs):       
                j = 0
                for (c,i) in zip(coeffs, indices):
                    i += len(self.conditions())
                    ci = int(self.cimap[i])
                    mat[ci+j, j] = c
            return mat

        # otherwise, find a common hrf to the specified conditions
        list_hrf= [self._conditions[self.terms()[i]]['hrf model'] for i in indices]
        def _common_to_list(list_hrf):
            lhrf = list_hrf[0]
            for i in range(1,len(list_hrf)):
                lhi = list_hrf[i]
                lhrf = [h for h in lhrf if h in lhi]
            return lhrf
        lhrf = _common_to_list(list_hrf)
        
        if isinstance (hrf_term,str):
            hrf_term = [hrf_term]

        if hrf_term!=None:
            lhrf = [h for h in lhrf if h in hrf_term]
            if len(lhrf)<len(hrf_term):
                print 'warning: contrast implemented only for a subset of prescribed hrfs'
            
        if lhrf==None:
            raise ValueError, "the specified conditions have no common hrf term"
        
        nhrfs = len(lhrf)
        mat = np.zeros([nregressors, nhrfs])

        for (c,i) in zip(coeffs, indices):
            t = self.terms()[i]
            ch = self._conditions[t]['hrf model']
            aux = [h in lhrf for h in ch]
            J = np.nonzero(np.array(aux))[0]
            ci = int(self.cimap[i])
            mat[ci+J, J] = c
        
        return mat
