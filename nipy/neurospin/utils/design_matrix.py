"""
fMRI Design Matrix creation functions.
"""

import numpy as np
from nipy.modalities.fmri import formula, utils, hrf

class DesignMatrix(object):
    """
    Class to handle design matrices
    """

    def __init__(self, frametimes, paradigm=None, hrf_model='Canonical',
               drift_model='Cosine', hfcut=128, drift_order=1, fir_delays=[0],
               fir_duration=1., cond_ids=None, add_regs=None, add_reg_names=None):
        """
        Parameters
        ----------
        frametimes: array of shape(nbframes),
                    the timing of the scans
        paradigm: array of shape (nevents, nc)
                  paradigm-encoding array
                  if nc==2, the type is event-related design (condition id, onset)
                  if nc==3, it represents(condition id, onset, duration)
                  if nc==4, (condition id, onset, duration, intensity)
                  if nc>2 this is a block design, 
                     unless all durations are equal to 0
                  if paradigm==None, then no condition is included
        hrf_model, string, optional,
                   that specifies the hemodynamic reponse function
                   it can be 'Canonical', 'Canonical With Derivative' or 'FIR'
        drift_model, string 
                     specifies the desired drift model,
                     to be chosen among 'Polynomial', 'Cosine', 'Blank'
        hfcut=128:  float , 
                    cut frequency of the low-pass filter
        drift_order=1, int, 
                       order of the dirft model (in case it is polynomial) 
        fir_delays=[0], optional, 
                        array of shape(nb_onsets) or list
                        in case of FIR design, yields the array of delays 
                        used in the FIR model
        fir_duration=1., float, duration of the FIR block; 
                         in general it should be equal to the tr    
        cond_ids=None, list of strin of length (ncond), 
                       ids of the experimental conditions. 
                       If None this will be called 'c0',..,'cn'
        add_regs=None, array of shape(naddreg, nbframes)
                       additional user-supplied regressors
        add_reg_names=None: list of (naddreg) regressor names
                            if None, while naddreg>0, these will be termed
                            'reg_%i',i=0..naddreg-1
        path=None, if not None, 
                   the matrix is written as a .csv file at the given path
        """
        self.frametimes = frametimes
        self.paradigm = paradigm
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.hfcut = hfcut
        self.drift_order = drift_order
        self.fir_delays = fir_delays
        self.fir_duration = fir_duration
        self.cond_ids = cond_ids
        self.estimated = False  

        # todo : check arguments       
        if add_regs==None:
            self.n_add_regs = 0
        else:
            # check that regressor specification is correct
            if add_regs.shape[0] == np.size(add_regs):
                add_regs = np.reshape(add_regs, (np.size(1, add_regs)))
            if add_regs.shape[0] != np.size(frametimes):
                raise ValueError, 'incorrect specification of additional regressors'
            self.n_add_regs = add_regs.shape[1]
        self.add_regs = add_regs
        
        
        if  add_reg_names == None:
            self.add_reg_names = ['reg%d'%k for k in range(self.n_add_regs)]
        elif len(add_reg_names)!= self.n_add_regs:
             raise ValueError, 'Incorrect number of additional regressors names \
                               was provided'
        else: 
            self.add_reg_names = add_reg_names


    def estimate(self):
        """
        """
        self.drift = set_drift(self.drift_model, self.frametimes,
                               self.drift_order, self.hfcut)
        if self.paradigm==None:
            self.n_conditions = 0
            self.n_main_regressors = 0
            self.formula = self.drift
            self.names = []
        else:
            self.n_conditions = int(self.paradigm[:,0].max()+1)
            if self.cond_ids==None:
                self.cond_ids = ['c%d'%k for k in range(self.n_conditions)]
            self.conditions, self.names =\
                convolve_regressors(self.paradigm, self.hrf_model, self.cond_ids,
                                    self.fir_delays, self.fir_duration)
            self.n_main_regressors = len(self.names)
            self.formula = self.conditions + self.drift

        # sample the matrix    
        self.matrix = build_dmtx(self.formula, self.frametimes).T
        # FIXME: ugly  workaround the fact that NaN can occur when trials
        # are earlier than scans (!)
        self.matrix[np.isnan(self.matrix)] = 0
    
        # add user-supplied regressors
        if self.n_add_regs>0:
            self.matrix = np.hstack((self.matrix[:,:self.n_main_regressors],
                                     self.add_regs,
                                     self.matrix[:,self.n_main_regressors:]))
        
            # add the corresponding names
            self.names = self.names + self.add_reg_names

        # Force the design matrix to be full rank at working precision
        self.matrix, self.design_cond = full_rank(self.matrix)
        
        # complete the names with the drift terms                               
        for k in range(len(self.drift.terms)-1):
            self.names.append('drift_%d'%(k+1))                            
        self.names.append('constant')
        self.estimated = True


    def write_csv(self, path):
        """
        write oneselfs as a csv
        
        Parameters
        ----------
        path: string, path of the resulting csv file           
        """
        import csv 
        if self.estimated==False:
           self.estimate()
        
        fid = open(path, "wb")
        writer = csv.writer(fid)
        writer.writerow(self.names)
        writer.writerows(self.matrix)
        fid.close()

    def read_from_csv(self, path):
        """
        load self.matrix and self.names  from a csv file
        Parameter
        ---------
        path: string,
            path of the .csv file that includes the matriox and related information

        fixme
        -----
        needs to check that this is coherent with the information of self ?
        """
        import csv
        csvfile = open(path)
        dialect = csv.Sniffer().sniff(csvfile.read())
        csvfile.seek(0)
        reader = csv.reader(open(path, "rb"),dialect)
    
        boolfirst = True
        design = []
        for row in reader:
            if boolfirst:
                names = [row[j] for j in range(len(row))]
                boolfirst = False
            else:
                design.append([row[j] for j in range(len(row))])
                
            x = np.array([[float(t) for t in xr] for xr in design])
        self.matrix = x
        self.names = names

    def show(self, rescale=True):
        """
        Vizualization of a design matrix

        Parameters
        ----------
        rescale= True: rescale for visualization or not

        Returns
        -------
        ax, figure handle
        
        """
        if self.estimated==False:
            self.estimate()
            
        x = self.matrix.copy()
        
        if rescale:
            x = x/np.sqrt(np.sum(x**2,0))

        import matplotlib.pylab as mp
        ax = mp.figure()
        mp.imshow(x, interpolation='Nearest', aspect='auto')
        mp.xlabel('conditions')
        mp.ylabel('scan number')
        if names!=None:
            mp.xticks(np.arange(len(names)),names,rotation=60,ha='right')
        
        mp.subplots_adjust(top=0.99,bottom=0.25)
        return ax
        
def dmtx_light(frametimes, paradigm=None, hrf_model='Canonical',
               drift_model='Cosine', hfcut=128, drift_order=1, fir_delays=[0],
               fir_duration=1., cond_ids=None, add_regs=None, add_reg_names=None,
               path=None):
    """
    Make a design matrix while avoiding framework
    
    Parameters
    ----------
    frametimes: array of shape(nbframes),
        the timing of the scans
    paradigm: array of shape (nevents, nc)
             paradigm-encoding array
             if nc==2, the type is event-related design (condition id, onset)
             if nc==3, it represents(condition id, onset, duration)
             if nc==4, (condition id, onset, duration, intensity)
             if nc>2 this is a block design, unless all durations are equal to 0
             if paradigm==None, then no condition is included
    hrf_model, string, optional,
             that specifies the hemodynamic reponse function
             it can be 'Canonical', 'Canonical With Derivative' or 'FIR'
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
    dmtx array of shape(nreg, nbframes):
        the sampled design matrix
    names list of strings of len (nreg)
        the names of the columns of the design matrix
    """
    DM = DesignMatrix(frametimes, paradigm, hrf_model, drift_model, hfcut,
                      drift_order, fir_delays, fir_duration, cond_ids,
                      add_regs, add_reg_names)
    DM.estimate()
    if path is not None:
        DM.write_csv(path)
    return DM.matrix, DM.names

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
    DriftModel: string, 
                to be chosen among 'Polynomial', 'Cosine', 'Blank'
                that specifies the desired drift model
    frametimes: array of shape(ntimes),
                list of values representing the desired TRs
    order: int, optional,
        order of the dirft model (in case it is polynomial)
    hfcut: float, optional,
        frequency cut in case of a cosine model

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
    paradigm: array of shape (nevents, nc), optional
             paradigm-encoding array
             if nc==2, the type is event-related design (condition id, onset)
             if nc==3, it represents(condition id, onset, duration)
             if nc==4, (condition id, onset, duration, intensity)
             if nc>2 this is a block design, unless all durations are equal to 0
             if paradigm==None, then no condition is included
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
    typep='block'
    if paradigm.shape[1]==2:
        typep = 'event'  
    elif paradigm[:,2].max()==0:
        typep='event'
 
    for nc in range(ncond):
        onsets =  paradigm[paradigm[:,0]==nc, 1]
        nos = np.size(onsets)
        if paradigm.shape[1]==4:
            values = paradigm[paradigm[:,0]==nc, 3]
        else:
            values = np.ones(nos)
        if nos>0:
            if typep=='event':
                if hrf_model=="Canonical":
                    c = formula.define(names[nc],
                                       utils.events(onsets, values, f=hrf.glover))
                    listc.append(c)
                    hnames.append(names[nc])
                elif hrf_model=="Canonical With Derivative":
                    c1 = formula.define(names[nc],
                                        utils.events(onsets, values, f=hrf.glover))
                    c2 = formula.define(names[nc]+"_derivative",
                                        utils.events(onsets, values, f=hrf.dglover))
                    listc.append(c1)
                    listc.append(c2)
                    hnames.append(names[nc])
                    hnames.append(names[nc]+"_derivative")
                elif hrf_model=="FIR":
                    for i,ft in enumerate(fir_delays):
                        lnames = names[nc] + "_delay_%d"%i
                        changes = np.hstack((onsets+ft, onsets+ft+fir_duration))
                        ochanges = np.argsort(changes)
                        lvalues = np.hstack((values, np.zeros(nos)))
                        changes = changes[ochanges]
                        lvalues = lvalues[ochanges]
                        
                        c = formula.define(lnames, utils.step_function(changes, lvalues))

                        listc.append(c)
                        hnames.append(lnames)
                else:
                    raise NotImplementedError,'unknown hrf model'
            elif typep=='block':
                offsets =  onsets + paradigm[paradigm[:,0]==nc,2]
                changes = np.hstack((onsets, offsets))
                values = np.hstack((values, -values))

                if hrf_model=="Canonical":
                    c = utils.events(changes, values, f=hrf.iglover)
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
