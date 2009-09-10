
__doc__ = """
fMRI Design Matrix creation functions.
"""

import numpy as np
from nipy.modalities.fmri import formula, utils, hrf


def _trial_dmtx():
    """ test code to make a design matrix
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    conditions = [0,0,0,1,1,1,3,3,3]
    onsets=[30,70,100,10,30,90,30,40,60]
    paradigm = np.vstack(([conditions, onsets])).T
    hrf_model = 'Canonical'
    X,names = dmtx_light(frametimes, paradigm, drift_model='Polynomial', order=3)
    import matplotlib.pylab as mp
    mp.figure()
    mp.imshow(X/np.sqrt(np.sum(X**2,0)),interpolation='Nearest')
    mp.show()
    print names

def _trial_dmtx_block():
    """ test code to make a design matrix
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    conditions = [0,0,0,1,1,1,3,3,3]
    onsets=[30,70,100,10,30,90,30,40,60]
    duration = 6*np.ones(9)
    paradigm = np.vstack(([conditions, onsets, duration])).T
    hrf_model = 'Canonical'
    X,names = dmtx_light(frametimes, paradigm, drift_model='Polynomial', order=3)
    import matplotlib.pylab as mp
    mp.figure()
    mp.imshow(X/np.sqrt(np.sum(X**2,0)),interpolation='Nearest')
    mp.show()
    print names
    
def dmtx_light(frametimes, paradigm, hrf_model='Canonical',
               drift_model='Cosine', hfcut=128, order=1, names=None):
    """
    Light-weight function to make easily a design matrix while avoiding framework
    
    Parameters
    ----------
    frametimes, array of shape(nbframes) the timing of the scans
    paradigm array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that constains (condition id, onset) or (condition id, onset, duration)
    hrf_model, string that can be 'Canonical', 'Canonical With Derivative' or 'FIR'
               that specifies the hemodynamic reponse function
    drift_model, string that specifies the desired drift model,
                to be chosen among 'Polynomial', 'Cosine', 'Blank'
    hfcut=128  float , cut frequency of the low-pass filter
    order=1, int, order of the dirft model (in case it is polynomial) 
    names=None, list of strin of length (ncond), ids of the experimental conditions. 
                If None this will be called 'c0',..,'cn'
    
    Returns
    -------
    dmtx array of shape(ncond, nbframes): the sampled design matrix
    names list of trings; the names of the columns of the design matrix
    """
    if names==None:
        names = ['c%d'%k for k in range(int(paradigm[:,0].max()+1))]
    drift = set_drift(drift_model, frametimes, order, hfcut)
    conditions, names = convolve_regressors(paradigm, hrf_model, names)
    formula = conditions + drift
    dmtx = build_dmtx(formula, frametimes).T
    
    # fixme : ugly  workaround the fact that NaN can occu when trials
    # are earlier than scans (!)
    dmtx[np.isnan(dmtx)] = 0
    ## Force the design matrix to be full rank at working precision
    dmtx, design_cond = fullRank(dmtx)
        
    # complete the names with the drift terms                               
    for k in range(len(drift.terms)-1):
        names.append('drift_%d'%(k+1))                            
    names.append('constant')
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


def convolve_regressors(paradigm, hrf_model, names=None):
    """
    Creation of  a formula that represents 
    the convolution of the conditions onset witha  certain hrf model
    
    Parameters
    ----------
    paradigm array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that constains (condition id, onset) or 
             (condition id, onset, duration)
    hrf_model, string that can be 'Canonical', 
               'Canonical With Derivative' or 'FIR'
               that specifies the hemodynamic reponse function
    names=None, list of strings corresponding to the condition names
                if names==None, these are create as 'c1',..,'cn'
                meaning 'condition 1'.. 'condition n'
    
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
    not sure how to deal with absent regressors
    FIR design not implemented yet
    normalization of the hrf
    """
    ncond = int(paradigm[:,0].max()+1)
    if names==None:
        names=["c%d"%k for  k in ncond]
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


def fullRank(X, cmax=1e15):
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
