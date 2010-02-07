"""
fMRI Design Matrix creation functions.
"""

import numpy as np

from nipy.modalities.fmri import formula, utils, hrf

    
def dmtx_light(frametimes, paradigm=None, hrf_model='Canonical',
               drift_model='Cosine', hfcut=128, drift_order=1, fir_delays=[0],
               fir_duration=1., cond_ids=None, add_regs=None, add_reg_names=None,
               path=None):
    """
    Make a design matrix while avoiding framework
    
    Parameters
    ----------
    frametimes, array of shape(nbframes) the timing of the scans
    paradigm=None, array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that constains (condition id, onset) or (condition id, onset, duration)
             if paradigm==None, then no conditions are included
    hrf_model, string that can be 'Canonical', 'Canonical With Derivative' or 'FIR'
               that specifies the hemodynamic reponse function
    drift_model, string that specifies the desired drift model,
                to be chosen among 'Polynomial', 'Cosine', 'Blank'
    hfcut=128  float , cut frequency of the low-pass filter
    drift_order=1, int, order of the dirft model (in case it is polynomial) 
    fir_delays=[0], optional, array of shape(nb_onsets) or list
                    in case of FIR design, yields the array of delays 
                    used in the FIR model
    fir_duration=1., float, duration of the FIR block; 
                     in general it should be equal to the tr    
    cond_ids=None, list of strin of length (ncond), 
                     ids of the experimental conditions. 
                If None this will be called 'c0',..,'cn'
    add_regs=None, array of shape(naddreg, nbframes)
                   additional user-supplied regressors
    add_reg_names=None, list of (naddreg) regressor names
                        if None, while naddreg>0, these will be termed
                        'reg_%i',i=0..naddreg-1
    path=None, if not None, the matrix is written as a .csv file at the given path
                 
    Returns
    -------
    dmtx array of shape(nreg, nbframes): the sampled design matrix
    names list of trings; the names of the columns of the design matrix
    """
    drift = set_drift(drift_model, frametimes, drift_order, hfcut)
    if paradigm==None:
       formula = drift
       names = []
    else:
        if cond_ids==None:
           cond_ids = ['c%d'%k for k in range(int(paradigm[:,0].max()+1))]
           conditions, names = convolve_regressors(paradigm, hrf_model, 
                                        cond_ids, fir_delays, fir_duration)
           formula = conditions + drift
    dmtx = build_dmtx(formula, frametimes).T
    
    # FIXME: ugly  workaround the fact that NaN can occur when trials
    # are earlier than scans (!)
    dmtx[np.isnan(dmtx)] = 0
    
    # add user-supplied regressors
    if add_regs!=None:
        # check that regressor specification is correct
        if add_regs.shape[0] == np.size(add_regs):
            add_regs = np.reshape(add_regs, (np.size(1, add_regs)))
        if add_regs.shape[0] != np.size(frametimes):
           raise ValueError, 'incorrect specification of additional regressors'
        
        # put them at the right place in the dmtx
        ncr = len(names)
        dmtx = np.hstack((dmtx[:,:ncr], add_regs, dmtx[:,ncr:]))
        
        # add the corresponding names
        if  add_reg_names == None:
            add_reg_names = ['reg%d'%k for k in range(add_regs.shape[1])]
        elif len(add_reg_names)!= add_regs.shape[1]:
             raise ValueError, 'Incorrect number of additional regressors names \
                               was provided'
        for r in range(add_regs.shape[1]):
            names.append(add_reg_names[r])

    # Force the design matrix to be full rank at working precision
    dmtx, design_cond = full_rank(dmtx)
        
    # complete the names with the drift terms                               
    for k in range(len(drift.terms)-1):
        names.append('drift_%d'%(k+1))                            
    names.append('constant')

    # possibly write the result
    if path is not None:
        import csv
        writer = csv.writer(open(path, "wb"))
        writer.writerow(names)
        writer.writerows(dmtx)
    return dmtx, names


def _polydrift(order, tmax):
    """
    Create a polynomial drift formula
    
    Parameters
    ----------
    order, int, number of polynomials in the drift model
    tmax, float maximal time value used in the sequence
          this is used to normalize porperly the columns
    
    Returns
    -------
    pol a formula that contains all the polynomial drift 
    plus a constant regressor
    """
    t = formula.Term('t')
    pt = []
    # fixme : ideally  this should be orthonormalized  
    for k in range(order):
        pt.append(formula.define('poly_drift_%d'%(k+1),t**(k+1)/tmax**(k+1))) 
    pt.append(formula.define('constant',1.0+0*t))
    pol =  formula.Formula(pt)
    return pol


def _cosinedrift(hfcut, tmax, tsteps):
    """
    Create a cosine drift formula

    Parameters
    ----------
    hfcut, float , cut frequency of the low-pass filter
    tmax, float  maximal time value used in the sequence
    tsteps, int,  number of TRs in the sequence
    
    Returns
    -------
    cos  a formula that contains all the polynomial drift 
    plus a constant regressor
    """
    t = formula.Term('t')
    pt = []
    order = int(np.floor(2 * float(tmax) / float(hfcut)) + 1)
    for k in range(1,order):
        u = np.sqrt(2.0/tmax) * utils.sympy_cos(np.pi*(t/tmax+ 0.5/tsteps)*k )
        pt.append(formula.define('cosine_drift_%d'%(k+1),u)) 
    pt.append(formula.define('constant',1.0+0*t))
    cos =  formula.Formula(pt)
    return cos


def _blankdrift():
    """
    Create the blank drift formula
    
    Returns
    -------
    df  a formula that contains a constant regressor
    """
    t = formula.Term('t')
    pt = [formula.define('constant',1.0+0*t)]
    df =  formula.Formula(pt)
    return df


def set_drift(DriftModel, frametimes, order=1, hfcut=128.):
    """
    Create the drift formula
    
    Parameters
    ----------
    DriftModel, string that specifies the desired drift model,
                to be chosen among "Polynomial", "Cosine", "Blank"
    frametimes array of shape(ntimes),
                list of values representing the desired TRs
    order=1, int, order of the dirft model (in case it is polynomial)
    hfcut=128., float, frequency cut in case of a cosine model

    Returns
    -------
    df, the resulting drift formula
    """
    if DriftModel=='Polynomial':
        d = _polydrift(order, frametimes.max())
    elif DriftModel=='Cosine':
        d = _cosinedrift(hfcut, frametimes.max(), frametimes.size)
    elif DriftModel=='Blank':
        d  = _blankdrift()
    else:
        raise NotImplementedError,"unknown drift model"
    return d


def convolve_regressors(paradigm, hrf_model, names=None, fir_delays=[0], 
    fir_duration = 1.):
    """
    Creation of  a formula that represents 
    the convolution of the conditions onset witha  certain hrf model
    
    Parameters
    ----------
    paradigm array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that contains (condition id, onset) or 
             (condition id, onset, duration)
    hrf_model, string that can be 'Canonical', 
               'Canonical With Derivative' or 'FIR'
               that specifies the hemodynamic reponse function
    names=None, list of strings corresponding to the condition names
                if names==None, these are create as 'c1',..,'cn'
                meaning 'condition 1'.. 'condition n'
    fir_delays=[0], optional, array of shape(nb_onsets) or list
                    in case of FIR design, yields the array of delays 
                    used in the FIR model
    fir_duration=1., float, duration of the FIR block; 
                     in general it should eb equal to the tr    
 
    Returns
    -------
    f a formula object that contains the convolved regressors 
      as functions of time    
    names list of strings corresponding to the condition names
          the output names depend on teh hrf model used
          if 'Canonical' then this is identical to the input names
          if 'Canonical With Derivative', then two names are produced for
             input name 'name': 'name' and 'name_derivative'

    fixme: 
    normalization of the columns of the design matrix ?
    """
    paradigm = np.asarray(paradigm)
    if paradigm.ndim !=2:
        raise ValueError('Paradigm should have 2 dimensions')
    ncond = int(paradigm[:,0].max()+1)
    if names==None:
        names=["c%d" % k for k in range(ncond)]
    else:
        if len(names)<ncond:
            raise ValueError, 'the number of names is less than the \
                  number of conditions'   
        else:
            ncond = len(names)        
    listc = []
    hnames = []
    if paradigm.shape[1]>2:
        typep = 'block'  
    else:
        typep='event'
 
    for nc in range(ncond):
        onsets =  paradigm[paradigm[:,0]==nc,1]
        nos = np.size(onsets) 
        if nos>0:
            if typep=='event':
                if hrf_model=="Canonical":
                    c = formula.define(names[nc], utils.events(onsets, f=hrf.glover))
                    listc.append(c)
                    hnames.append(names[nc])
                elif hrf_model=="Canonical With Derivative":
                    c1 = formula.define(names[nc],
                                        utils.events(onsets, f=hrf.glover))
                    c2 = formula.define(names[nc]+"_derivative",
                                        utils.events(onsets, f=hrf.dglover))
                    listc.append(c1)
                    listc.append(c2)
                    hnames.append(names[nc])
                    hnames.append(names[nc]+"_derivative")
                elif hrf_model=="FIR":
                    for i,ft in enumerate(fir_delays):
                        lnames = names[nc]+"_delay_%d"%i
                        changes = np.hstack((onsets+ft,onsets+ft+fir_duration))
                        ochanges = np.argsort(changes)
                        values = np.hstack((np.ones(nos), np.zeros(nos)))
                        changes = changes[ochanges]
                        values = values[ochanges]
                        c = formula.define(lnames, utils.step_function(changes,values))
                        listc.append(c)
                        hnames.append(lnames)
                else:
                    raise NotImplementedError,'unknown hrf model'
            elif typep=='block':
                offsets =  onsets+paradigm[paradigm[:,0]==nc,2]
                changes = np.hstack((onsets,offsets))
                values = np.hstack((np.ones(nos), -np.ones(nos)))

                if hrf_model=="Canonical":
                    c = utils.events(changes,values, f=hrf.iglover)
                    listc.append(c)
                    hnames.append(names[nc])
                elif hrf_model=="Canonical With Derivative":
                    c1 = utils.events(changes,values, f=hrf.iglover)
                    c2 = utils.events(changes,values, f=hrf.glover)
                    listc.append(c1)
                    listc.append(c2)
                    hnames.append(names[nc])
                    hnames.append(names[nc]+"_derivative")
                elif hrf_model=="FIR":
                    raise NotImplementedError,\
                          'block design are not compatible with FIR at the moment'
                else:
                    raise NotImplementedError,'unknown hrf model'  
            else:
                raise NotImplementedError,'unknown type of paradigm'
    
    # create the formula
    p = formula.Formula(listc)
     
    return p, hnames


def build_dmtx(form, frametimes):
    """
    This is a work arount to control the order of the regressor 
    in the design matrix construction

    Parameters
    ----------
    form, the formula that describes the design matrix
    frametimes, array of shape (nb_time_samples), the time sampling grid

    Returns
    -------     
    X array of shape (nrows,nb_time_samples) the sdesign matrix
    """
    # fixme : workaround to control matrix columns order
    t = formula.make_recarray(frametimes, 't')
    X = []
    for ft in form.terms:
        lf = formula.Formula([ft])
        X.append(lf.design(t, return_float=True))
    X = np.array(X)
    return X 


def full_rank(X, cmax=1e15):
    """
    This function possibly adds a scalar matrix to X
    to guarantee that the condition number is smaller than a given threshold.

    Parameters
    ----------
    X array of shape(nrows,ncols)
    cmax=1.e-15, float tolerance for condition number

    Returns
    -------
    X array of shape(nrows,ncols) after regularization
    cmax=1.e-15, float tolerance for condition number
    """
    U, s, V = np.linalg.svd(X,0)
    sM = s.max()
    sm = s.min()
    c = sM/sm
    if c < cmax:
        return X, c
    print 'Warning: matrix is singular at working precision, regularizing...'
    lda = (sM-cmax*sm)/(cmax-1)
    s = s + lda
    X = np.dot(U, np.dot(np.diag(s), V))
    return X, cmax
