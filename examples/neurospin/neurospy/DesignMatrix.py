__doc__ = """
fMRI Design Matrix creation functions.
"""

import os, urllib, time, string
import numpy as np
import pylab
from configobj import ConfigObj
from nipy.modalities.fmri import formula, utils, hrf

def _trial():
    """
    just for debugging
    """
    import sympy
    tr = 1.0
    c1 = formula.define('condition1', utils.events([30,70,100], f=hrf.glover))
    #c2 = formula.define('condition2', utils.events([10,30,90], f=hrf.glover))
    c2 = formula.define('condition2', utils.events([30,30.1,70,70.1,100,100.1],
                                                          [1,-1,1,-1,1, -1], f=hrf.iglover))
    #c3 = formula.define('condition3', utils.events([30,40,60], f=hrf.glover))
    c3 = formula.define('condition3', utils.events([30,45,90,105,60,75],
                                                          [1,-1,1,-1,1, -1], f=hrf.iglover))
    fb = np.array([1,2],np.float)/(128*tr)
    d = utils.fourier_basis(fb)

    t = formula.Term('t')
    pt = []
    for k in range(3):
        pt.append(formula.define('poldrift%d'%k,t**(k+1))) 
    
    pol =  formula.Formula(pt)
    
    f = formula.Formula([c1,c2,c3]) + pol
    t = formula.make_recarray(np.linspace(0,127*tr,128), 't')
    
    X = f.design(t, return_float=True)

    import matplotlib.pylab as mp
    mp.figure()
    mp.imshow(X/np.std(X,0),interpolation='Nearest')
    mp.show()
    return X,f

def _trial_dmtx():
    """ test code to make a design matrix
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    conditions = [0,0,0,1,1,1,2,2,2]
    onsets=[30,70,100,10,30,90,30,40,60]
    paradigm = np.vstack(([conditions],[onsets])).T
    hrf_model='Canonical'
    X,names= dmtx_light(frametimes, paradigm, drift_model='Polynomial', order=3)
    import matplotlib.pylab as mp
    mp.figure()
    mp.imshow(X/np.sqrt(np.sum(X**2,0)),interpolation='Nearest')
    mp.show()
    print names


def dmtx_light(frametimes, paradigm, hrf_model='Canonical', drift_model='Cosine', hfcut=128,
               order=1, names=None):
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
        names = ['c%d'%k for k in range(paradigm[:,0].max()+1)]
    drift = _set_drift(drift_model, frametimes, order, hfcut)
    conditions, cnames = _convolve_regressors(paradigm, hrf_model, names)
    formula = conditions + drift
    dmtx = _build_dmtx(formula, frametimes).T

    ## Force the design matrix to be full rank at working precision
    dmtx, design_cond = _fullRank(dmtx)
        
    # complete the names with the drift terms                               
    for k in range(len(drift.terms)-1):
        names.append('poly_drift_%d'%(k+1))                            
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
    pol a formula that contains all teh polynomial drift plus a constant regressor
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
    cos  a formula that contains all the polynomial drift plus a constant regressor
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

def _set_drift(DriftModel, frametimes, order=1, hfcut=128.):
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

def _loadProtocol(path, session):
    """
    Read a paradigm file consisting of a list of pairs
    (occurence time, (duration), event ID)
    and create a paradigm array
    
    Parameters
    ----------
    path, string a path to a .csv file that describes the paradigm
    session, int, the session number used to extract 
             the relevant session information in th csv file
    
    Returns
    -------
    paradigm array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that constains (condition id, onset) or (condition id, onset, duration)
    """
    paradigm = pylab.loadtxt(path)
    if paradigm[paradigm[:,0] == session].tolist() == []:
        return None
    paradigm = paradigm[paradigm[:,0] == session]
    
    if paradigm.shape[1] == 4:
        paradigm = paradigm[:,1:]
        typep = 'block'
    else:
        typep ='event'
        paradigm = paradigm[:,[1,2]]
    
    return paradigm

def _convolve_regressors(paradigm, hrf_model, names=None):
    """
    Creation of  a formula that represents 
    the convolution of the conditions onset witha  certain hrf model
    
    Parameters
    ----------
    paradigm array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that constains (condition id, onset) or (condition id, onset, duration)
    hrf_model, string that can be 'Canonical', 'Canonical With Derivative' or 'FIR'
               that specifies the hemodynamic reponse function
    
    fixme: 
    fails if one of the conditions is not present
    FIR design not tmplmented yet
    """
    # fixme: what should we do if names==None ?
    # fixme : add the FIR design
    ncond = int(paradigm[:,0].max()+1)
    listc = []
    hnames = []
    if paradigm.shape[1]>2:
        typep = 'block'  
    else:
        typep='event'

    for nc in range(ncond):
        onsets =  paradigm[paradigm[:,0]==nc,1]
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
            offsets =  paradigm[paradigm[:,0]==nc,2]
            changes = np.hstack((onsets,offsets))
            values = np.hstack((np.ones(np.size(onsets)), np.ones(np.size(offsets))))

            if hrf_model=="Canonical":
                c = utils.events(onsets,values, f=hrf.iglover)
                listc.append(c)
                hnames.append(names[nc])
            elif hrf_model=="Canonical With Derivative":
                c1 = utils.events(onsets,values, f=hrf.iglover)
                c2 = utils.events(onsets,values, f=hrf.glover)
                listc.append(c1)
                listc.append(c2)
                hnames.append(names[nc])
                hnames.append(names[nc]+"_derivative")
            else:
                raise NotImplementedError,'unknown hrf model'  

            #if names != None:
            #    c = formula.define(names[nc], c)
            #hnames.append("condition %d",nc)
            #listc.append(c)
                
                
        else:
            raise NotImplementedError,'unknown type of paradigm'
    p = formula.Formula(listc)
            
    # fixme : how to handle blocks
    return p, hnames

def _build_dmtx(form, frametimes):
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



class DesignMatrix():
    """
    Design mtrices handling class
    class members are:
    protocol_filename: string, path of the protocole-defining file
    nbframes, int, the number of volumes corresponding to the session
    session, int or string, a session id
    misc_file, string, a path to a file that stores some information on conditions
    model, "default": this is a brainvisa thing
    frametimes, array of shape(nbframes): time stamps of the scans
    names: names of the differnt entries of the design matrix
    protocol: array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that constains (condition id, onset) or (condition id, onset, duration)
    """
    def __init__(self, nbvols, protocol_filename, session = 0,
                 misc_file=None, model="default"):
        self.protocol_filename = protocol_filename
        self.nbframes = nbvols
        self.session = session
        self.misc_file = misc_file
        self.model = model

    def load(self):
        """
        Load data from files and apply mask.  
        """
        
        if self.session.isdigit():
            self.session = int(self.session)
        else:
            misc = ConfigObj(self.misc_file)
            self.session = misc["sessions"].index(self.session)
       
        self.frametimes = np.arange(self.nbframes)
        self.misc = ConfigObj(self.misc_file)
        if not self.misc.has_key(self.model):
            misc[self.model] = {}
        misc.write()

        # fixme id self.misc["tasks"]==None
        self._names = self.misc["tasks"]
        self.protocol = _loadProtocol(self.protocol_filename, self.session)
    
    def timing(self, tr, t0=0.0, trSlices=None, slice_idx=None):
        """
        Parameters
        ----------
        tr : inter-scan repetition time,
           i.e. the time elapsed between two consecutive scans
        t0 : time elapsed from the paradigm time origin 
             to the first scan acquisition 
             (different from zero if the paradigm was not synchronized 
             with the acquisition, or dummy scans have been removed)
        trSlices : inter-slice repetition time, same concept as tr for slices  
        slice_idx : either a string or an array of integers.      
                  When input as an array of integers, 
                  slice_idx is the slice acquisition order that 
                  maps each slice number to its corresponding rank 
                  (be careful, indexes are counted from zero instead of one, 
                  as it is the standard practice in Python). 
                  By convention, slices are numbered from the bottom to the top 
                  of the head. 
                  Alternatively, keywords describing
                  usual sequences can be used:
                  'ascending'  : equivalent to [0,1,2,...,N-1]
                  'descending' : equivalent to [N-1,N-2,...,0] 
                  'interleaved bottom-up' : equivalent to [0,N/2,1,N/2+1,2,N/2+2,...]
                  'interleaved top-down' : reverted interleaved bottom-up 
        """
        tr = float(tr)
        t0 = float(t0)
        self.frametimes *= tr
        self.frametimes += t0

    def set_drift(self, DriftModel="Polynomial", order=1, hfcut=128):
        """
        Set the drift formula of the model
     
        Parameters
        ----------
        DriftModel, string that specifies the desired drift model,
                    to be chosen among "Polynomial", "Cosine", "Blank"
        order=1, int, order of the dirft model (in case it is polynomial)
        hfcut=128., float, frequency cut in case of a cosine model
        """
        self.drift = _set_drift(DriftModel, self.frametimes, order, hfcut)

    def set_conditions(self, hrfmodel="Canonical"):
        """ 
        Set the conditions of the model as a formula
        
        """
        if self.protocol == None:
           self.conditions = None
        else:
            self.conditions, self._names = _convolve_regressors(self.protocol,
                                                                hrfmodel, self._names)
         
    def compute_design(self, name="", verbose=1):
        """Sample the formula on the grid
        
        Note: self.conditions and self.drift must be defined beforhand
        """
        if self.protocol == None:
            print "The selected session does not exists"
            return None   
        self.formula = self.conditions + self.drift
        temp = _build_dmtx(self.formula, self.frametimes).T

        ## Force the design matrix to be full rank at working precision
        self._design, self._design_cond = _fullRank(temp)
        
        # complete the names with the drift terms                               
        for k in range(len(self.drift.terms)-1):
           self._names.append('poly_drift_%d'%(k+1))                            
        self._names.append('constant')

        if verbose:
           self.show()
        
        self.names = self._names
        misc = ConfigObj(self.misc_file)
        misc[self.model]["regressors_%s" % name] = self._names
        misc[self.model]["design matrix cond"] = self._design_cond
        misc.write()

    def show(self):
        """
        """
        X = self._design
        import matplotlib.pylab as mp
        mp.figure()
        mp.imshow(X/np.sqrt(np.sum(X**2,0)),interpolation='Nearest')

    def compute_fir_design(self, drift=None, o=1, l=1, name=""):
        """
        deprecated
        """
        if self.protocol == None:
            print "The selected session does not exists"
            return None
        misc = ConfigObj(self.misc_file)
        temp = np.zeros((len(self.frametimes), (o * len(self.protocol.events))))
        diff = l / o
        self.names = []
        i = 0
        for event in misc["tasks"]:
            if  self.protocol.events.has_key(event):
                for j in range(o):
                    if j == 0:
                        self.names.append("%s" % (event))
                    else:
                        self.names.append("%s_d%i" % (event, j))
                    for t in self.protocol.events[event].times:
                        base = np.argmax(self.frametimes > t)
                        for k in range(diff):
                            temp[base + (k + j * diff), j + i * o] = 1
                i += 1
        self._design, self._design_cond = _fullRank(temp)
        if drift == 0:
            drm = np.ones((self._design.shape[0],1))
        elif drift == cosine_drift:
            drm = cosine_matrix(self.frametimes).T
        elif drift == canonical_drift:
            drm = _drift(self.frametimes).T
        else:
            drm = drift
        drml = drm.shape[1]
        for i in range(drml):
            self.names.append('(drift:%i)' % i)
        self._design = np.column_stack((self._design, drm))
        misc[self.model]["regressors_%s" % name] = self.names
        misc.write()


def _fullRank(X, cmax=1e15):
    """ X is assumed to be a 2d numpy array. This function possibly adds a scalar matrix to X
    to guarantee that the condition number is smaller than a given threshold. """
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


